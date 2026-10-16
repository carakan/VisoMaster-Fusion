"""Optional torch/MPS backend for the ONNX face models.

Why this exists
---------------
Inference binds torch CPU tensors straight into ONNX Runtime through
``io_binding``. ONNX Runtime has no MPS execution provider, and a Metal buffer
pointer handed to the CPU/CoreML EPs is dereferenced as garbage rather than
failing loudly, so every model here runs on the CPU or CoreML provider. On this
machine that leaves the face swappers -- by far the most expensive models in the
pipeline -- running on the CPU.

Converting the ONNX graph to a torch module sidesteps the binding problem: torch
owns the memory, so the graph runs natively on MPS. ``TorchInferenceSession``
below mimics the small slice of the ``onnxruntime.InferenceSession`` surface this
codebase actually uses, including ``io_binding``, so no call site has to change.

Only some models move. This is not a blanket conversion -- measured on this
machine (i9-10900 / RX 6800 XT, per call, warmed):

    model                             ORT cpu   CoreML    torch MPS
    inswapper_128.fp16                    433 ms   526 ms        49 ms
    InStyleSwapper256_Version_A.fp16     2391 ms  2322 ms       147 ms
    alphaface_swapper_fused_norm         2430 ms   928 ms       130 ms
    ghost_unet_3_block                     309 ms   657 ms       161 ms
    det_10g                                39 ms    64 ms        22 ms
    yoloface_8n                            22 ms    41 ms        28 ms
    w600k_r50   (arcface embed)            35 ms     6 ms        23 ms
    simswap_arcface_model                  178 ms    54 ms        96 ms

The swappers gain 4-16x. The small embedders *lose* -- CoreML is already 3.8x
faster than MPS on w600k_r50, and roughly the same on the SimSwap embedder --
because at 112x112 the per-call MPS dispatch overhead is larger than the
arithmetic. ``MPS_MODELS`` below is therefore an allowlist, not a blocklist, so
adding a new model never silently routes it to a backend that is slower for it.

Turn it off with ``VISOMASTER_MPS_BACKEND=off``, or force it on with
``=torch`` (which raises if MPS or onnx2torch is unavailable). The default,
``auto``, uses torch only when MPS is present.
"""

from __future__ import annotations

import ctypes
import os
import threading

import numpy as np

_LOCK = threading.RLock()
_CACHE: dict = {}

_TORCH_STATE = None  # lazily resolved (torch, onnx_load, onnx_to_torch)

# Environment variable -> behaviour of resolve_backend().
_BACKEND_ENV = "VISOMASTER_MPS_BACKEND"

# Models whose ONNX file should run on MPS. Keyed by the file stem that
# ModelsProcessor receives, so this survives upstream renames of display names.
#
# Everything here was measured faster on MPS than on the CoreML provider this
# machine currently selects. Models absent from this list keep using ONNX
# Runtime -- see the module docstring for why that is the safe default.
MPS_MODELS = frozenset(
    {
        "inswapper_128",
        "inswapper_128.fp16",
        "InStyleSwapper256_Version_A",
        "InStyleSwapper256_Version_A.fp16",
        "InStyleSwapper256_Version_B",
        "InStyleSwapper256_Version_B.fp16",
        "InStyleSwapper256_Version_C",
        "InStyleSwapper256_Version_C.fp16",
        "alphaface_swapper_fused_norm",
        "alphaface_swapper_fused_norm.trtshape",
        "ghost_unet_1_block",
        "ghost_unet_3_block",
        "det_10g",
        # scrfd_2.5g_bnkps is deliberately NOT here. Measured on this machine:
        # ONNX Runtime CPU 22.0 ms vs onnx2torch MPS 35.5 ms. The model is small
        # enough that the host<->device copy and dispatch cost more than the
        # arithmetic, so MPS is a net loss -- and it is the default detector, so
        # it dominates the per-frame budget. Do not add it back without
        # re-measuring; the threshold is roughly "ORT CPU under ~50 ms".
    }
)


def _resolve_torch():
    """Import torch/onnx2torch once. Returns None when unavailable."""
    global _TORCH_STATE
    if _TORCH_STATE is None:
        try:
            import torch
            from onnx import load as onnx_load
            from onnx2torch import convert as onnx_to_torch

            _TORCH_STATE = (torch, onnx_load, onnx_to_torch)
        except Exception:
            _TORCH_STATE = False
    return _TORCH_STATE or None


def mps_available() -> bool:
    """True when torch can actually run on MPS."""
    state = _resolve_torch()
    if not state:
        return False
    torch = state[0]
    try:
        return bool(torch.backends.mps.is_available())
    except Exception:
        return False


def set_backend_enabled(enabled: bool) -> None:
    """Turn the backend on or off, following the provider selection.

    The provider combo box is the single control for this, and the app persists
    that choice in its own settings, so nothing is written here: keeping a second
    copy only produced a file that could disagree with the provider actually
    selected. ``VISOMASTER_MPS_BACKEND`` remains as a launch-time override for
    runs where the combo is not driven at all.

    The value is set in the environment rather than a module global because the
    backend is resolved per session at construction time, and the app reloads
    models after a provider change.
    """
    os.environ[_BACKEND_ENV] = "auto" if enabled else "off"


def resolve_backend() -> str:
    """Return "torch" or "onnx" after applying VISOMASTER_MPS_BACKEND."""
    choice = os.environ.get(_BACKEND_ENV, "auto").strip().lower()
    if choice in ("off", "0", "false", "no"):
        return "onnx"
    if choice in ("torch", "on", "1", "true", "yes"):
        if not _resolve_torch():
            raise RuntimeError(
                f"{_BACKEND_ENV}=torch but onnx2torch/torch could not be imported"
            )
        if not mps_available():
            raise RuntimeError(
                f"{_BACKEND_ENV}=torch but torch.backends.mps.is_available() is False"
            )
        return "torch"
    return "torch" if mps_available() else "onnx"


def should_use_mps(model_name: str, model_path: str = "") -> bool:
    """Whether this model is on the MPS allowlist and torch is usable.

    Accepts either the display name the app registers ("Inswapper128") or a path
    to the graph ("inswapper_128.fp16.onnx"); both are checked, because the
    display name and the file stem disagree for most swappers and a filename-only
    lookup silently matched nothing.
    """
    if resolve_backend() != "torch":
        return False
    candidates = {os.path.splitext(os.path.basename(model_name))[0]}
    if model_path:
        candidates.add(os.path.splitext(os.path.basename(model_path))[0])
    return bool(candidates & MPS_MODELS)


def _cache_key(path: str):
    try:
        st = os.stat(path)
        return (os.path.abspath(path), int(st.st_mtime), int(st.st_size))
    except OSError:
        return (os.path.abspath(path), 0, 0)


def _array_from_ptr(buffer_ptr: int, shape, element_type):
    """Rebuild a numpy view over memory another library owns.

    The call sites bind ``tensor.data_ptr()`` of a CPU torch tensor and pass the
    shape separately, so the bytes are already laid out contiguously in the
    right order -- all that is missing is a typed view.
    """
    shape = tuple(int(s) for s in shape)
    if not shape:
        return np.empty(0, dtype=element_type)
    # numpy 2 returns numpy.dtypes.* instances from np.dtype(), which have no
    # .dtype attribute, so normalise once and key the table on the dtype itself.
    element_type = np.dtype(element_type)
    ctype = {
        np.dtype(np.float32): ctypes.c_float,
        np.dtype(np.float16): ctypes.c_short,
        np.dtype(np.float64): ctypes.c_double,
        np.dtype(np.int64): ctypes.c_int64,
        np.dtype(np.int32): ctypes.c_int32,
        np.dtype(np.uint8): ctypes.c_ubyte,
    }[element_type]
    ptr = ctypes.cast(buffer_ptr, ctypes.POINTER(ctype))
    return np.ctypeslib.as_array(ptr, shape=shape)


class _TensorSpec:
    """Stand-in for onnxruntime's NodeArg, which is what the callers read."""

    __slots__ = ("name", "shape", "type")

    def __init__(self, name, shape, type_="tensor(float)"):
        self.name = name
        self.shape = shape
        self.type = type_


class TorchIOBinding:
    """Duck-typed replacement for ``ort_session.io_binding()``.

    Implements only what this codebase calls: ``bind_input``/``bind_output``
    with a raw pointer, the two clear methods, ``synchronize_outputs`` and
    ``copy_outputs_to_cpu``. Inputs stay bound until ``run`` consumes them, and
    outputs are materialised eagerly because ``copy_outputs_to_cpu`` is the only
    way callers read results.
    """

    def __init__(self, session: "TorchInferenceSession"):
        self._session = session
        self._inputs: list = []
        self._outputs: list = []
        self._output_buffers: dict = {}
        self._results: dict = {}

    def bind_input(self, name, device_type=None, device_id=None, element_type=None,
                   shape=None, buffer_ptr=None, **kwargs):
        if isinstance(element_type, str):
            element_type = np.dtype(element_type)
        elif element_type is not None:
            element_type = np.dtype(element_type)
        else:
            element_type = np.dtype(np.float32)
        arr = _array_from_ptr(buffer_ptr, shape, element_type)
        self._inputs.append((name, np.ascontiguousarray(arr)))
        return self

    def bind_cpu_input(self, name, input_or_arr, **kwargs):
        arr = input_or_arr
        if hasattr(arr, "numpy"):
            arr = arr.numpy()
        self._inputs.append((name, np.ascontiguousarray(arr)))
        return self

    def bind_output(self, name, device_type=None, device_id=None, buffer_ptr=None,
                     shape=None, element_type=None, **kwargs):
        self._outputs.append(name)
        # Some call sites (dfm_model.py) pre-allocate the output as a torch
        # tensor and pass its pointer, then read the result back out of that
        # tensor afterwards instead of calling copy_outputs_to_cpu. Record the
        # destination so run() can write into it.
        if buffer_ptr is not None:
            dtype = np.dtype(element_type) if element_type is not None else np.dtype(np.float32)
            dest = _array_from_ptr(buffer_ptr, shape, dtype) if shape else None
            self._output_buffers[name] = dest
        return self

    def bind_ortvalue_output(self, name, ortvalue, **kwargs):
        self._outputs.append(name)
        return self

    def clear_binding_inputs(self):
        self._inputs.clear()

    def clear_binding_outputs(self):
        self._outputs.clear()
        self._results.clear()

    def synchronize_outputs(self):
        torch = self._session._torch
        if self._session.device.type != "mps":
            return
        # run_ort_with_iobinding() calls this immediately after
        # run_with_iobinding(), from the worker thread, outside any backend lock.
        # An MPS synchronize racing another thread's submission aborts the
        # process with "commit an already committed command buffer", so it has to
        # be taken under the same lock as the inference itself.
        with _LOCK:
            torch.mps.synchronize()

    def run(self):
        """Execute the graph for whatever inputs are bound right now."""
        # No synchronize() here: run_outputs() already synchronises under the
        # backend lock. An extra MPS synchronize outside that lock is the same
        # cross-thread hazard this lock exists to prevent.
        results = self._session.run_outputs(self._inputs, self._outputs)
        # ORT materialises outputs lazily and copy_outputs_to_cpu() pulls them
        # out afterwards, so the values have to survive past this call.
        self._results = dict(zip(self._outputs, results))
        for name, dest in self._output_buffers.items():
            src = self._results.get(name)
            if src is None or dest is None:
                continue
            if src.shape == dest.shape:
                dest[...] = src
            else:
                # A pre-allocated buffer of a different shape means the caller
                # sized for a fixed layout; refuse rather than write garbage.
                raise ValueError(
                    f"output '{name}' shape {src.shape} does not match the bound "
                    f"buffer shape {dest.shape}"
                )
        return results

    def copy_outputs_to_cpu(self):
        if not self._results:
            self.run()
        return [self._results[name] for name in self._outputs if name in self._results]

    def get_outputs(self):
        return [self._results[name] for name in self._outputs if name in self._results]

    # ORT exposes get_cpu_outputs/get_device_outputs; callers here only use the
    # copy_ form, but keeping both avoids an AttributeError if that changes.
    def get_cpu_outputs(self):
        return self.copy_outputs_to_cpu()

    def get_device_outputs(self):
        return self.copy_outputs_to_cpu()


class TorchInferenceSession:
    """Drop-in for ``onnxruntime.InferenceSession`` for the allowlisted models.

    Mirrors the ORT API surface this codebase uses: ``get_inputs()``,
    ``get_outputs()``, ``run()`` and ``io_binding()``. Converted graphs are cached
    per ``(path, mtime, size)`` so repeatedly evicting and reloading a swapper
    does not pay the conversion each time.
    """

    def __init__(self, model_file, sess_options=None, providers=None, device=None):
        state = _resolve_torch()
        if not state:
            raise RuntimeError(
                "onnx2torch/torch unavailable: install them, or leave "
                f"{_BACKEND_ENV} unset"
            )
        torch, onnx_load, onnx_to_torch = state

        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found: {model_file}")

        self.model_file = model_file
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)
        self._torch = torch

        key = _cache_key(model_file)
        with _LOCK:
            cached = _CACHE.get(key)
        if cached is None:
            # Conversion reads the whole graph; a few seconds for a 256px swapper.
            graph = onnx_load(model_file)
            module = onnx_to_torch(graph).eval().to(self.device)
            with _LOCK:
                cached = _CACHE.get(key)
                if cached is None:
                    cached = (
                        module,
                        list(graph.graph.input),
                        list(graph.graph.output),
                    )
                    _CACHE[key] = cached
            del graph

        self._module, graph_inputs, graph_outputs = cached
        # The converted GraphModule's forward keeps ONNX input *order* but not
        # its names, so feeds are matched positionally against the graph order.
        self._input_specs = [
            _TensorSpec(
                i.name,
                [d.dim_value if d.dim_value else d.dim_param for d in i.type.tensor_type.shape.dim],
            )
            for i in graph_inputs
            if i.name
        ]
        self._output_specs = [_TensorSpec(o.name, []) for o in graph_outputs if o.name]
        self._graph_output_names = [o.name for o in graph_outputs if o.name]
        # Built on first failure, not up front: a healthy run never pays for an
        # extra ORT session it will not use.
        self._fallback = None

    # -- ORT surface ---------------------------------------------------------
    def get_inputs(self):
        return self._input_specs

    def get_outputs(self):
        return self._output_specs

    def io_binding(self):
        return TorchIOBinding(self)

    def run_with_iobinding(self, io_binding, run_options=None):
        """Execute a pre-populated binding.

        Every io_binding call site in this codebase funnels through
        ``FunctionWorker.run_ort_with_iobinding``, which calls this on the session
        and then ``io_binding.synchronize_outputs()`` -- it does not call
        ``io_binding.run()``. Without this method the session raises
        AttributeError and every model silently fails to process input.
        """
        if not isinstance(io_binding, TorchIOBinding):
            raise TypeError(
                "run_with_iobinding expects a binding created by this session's "
                "io_binding(); a binding from another backend cannot be executed"
            )
        return io_binding.run()

    def run(self, output_names, feed_dict, **kwargs):
        pairs = [(name, feed_dict[name]) for name in self._input_specs if name in feed_dict]
        # Callers that feed positionally by spec name are the norm here, but a
        # feed_dict keyed by graph order is also accepted.
        if len(pairs) != len(self._input_specs):
            pairs = [(self._input_specs[i].name, feed_dict[k])
                     for i, k in enumerate(feed_dict)]
        results = self.run_outputs(pairs, output_names or self._graph_output_names)
        return results

    # -- CPU fallback --------------------------------------------------------
    def _cpu_session(self):
        """A real ONNX Runtime CPU session for this graph, built on demand."""
        with _LOCK:
            if self._fallback is None:
                import onnxruntime as ort

                so = ort.SessionOptions()
                so.log_severity_level = 3
                self._fallback = ort.InferenceSession(
                    self.model_file, so, providers=["CPUExecutionProvider"]
                )
        return self._fallback

    def _run_cpu(self, pairs, output_names):
        """Run the original ONNX graph on the CPU, same contract as run_outputs."""
        feeds = {}
        for name, arr in pairs:
            a = arr.detach().cpu().numpy() if hasattr(arr, "detach") else np.asarray(arr)
            feeds[name] = np.ascontiguousarray(a)

        names = list(output_names) if output_names else list(self._graph_output_names)
        raw = self._cpu_session().run(names, feeds)
        return [r for r in raw]

    def _degrade(self, exc):
        """Record that this graph cannot run on MPS and explain why once."""
        with _LOCK:
            first = self.model_file not in _DEGRADED
            _DEGRADED.add(self.model_file)
        if first:
            name = os.path.basename(self.model_file)
            reason = f"{type(exc).__name__}: {str(exc).splitlines()[0][:110]}"
            print(
                f"[MPS Backend] {name} failed on MPS ({reason}). "
                "Falling back to ONNX Runtime on the CPU for this model; it will "
                "stay on the CPU until the app is restarted."
            )
            notifier = _FALLBACK_NOTIFIER
            if notifier is not None:
                try:
                    notifier(name, reason)
                except Exception:
                    # A notification problem must never take the render down.
                    pass

    # -- execution -----------------------------------------------------------
    def run_outputs(self, pairs, output_names):
        """Run the graph on MPS, or on the CPU if MPS cannot handle this graph.

        MPS is not uniformly supported: this torch build has whole kernel
        families that abort at runtime, and a converted graph may hit one that
        only a given input shape triggers. When that happens the model is moved
        to ONNX Runtime on the CPU rather than failing the frame, and it stays
        there -- the failure is a property of the graph, so retrying it every
        frame would pay the exception cost forever for no gain.

        This only covers exceptions MPS raises into Python. A Metal command
        buffer assertion aborts the process outright and cannot be caught here.
        """
        if _is_degraded(self.model_file):
            return self._run_cpu(pairs, output_names)

        try:
            return self._run_mps(pairs, output_names)
        except Exception as exc:
            self._degrade(exc)
            return self._run_cpu(pairs, output_names)

    def _run_mps(self, pairs, output_names):
        """Run the graph, transferring and collecting under a single lock.

        The whole transaction -- host->device copy, forward, synchronize and
        device->host copy -- has to be serialised. The app runs inference from
        several persistent worker threads (four by default), and MPS command
        buffers are shared per process: letting two threads submit copies at once
        aborts the whole process with

            -[MTLIOAccelCommandBuffer validate]:208: failed assertion
            `commit command buffer with uncommitted encoder'

        which is a hard crash, not a catchable exception. The previous version
        held the lock only across the forward pass and left the H2D copy outside
        it, which is exactly the window that triggered it.
        """
        torch = self._torch
        with _LOCK, torch.no_grad():
            tensors = []
            for _, arr in pairs:
                a = arr.detach().cpu().numpy() if hasattr(arr, "detach") else np.asarray(arr)
                tensors.append(
                    torch.from_numpy(np.ascontiguousarray(a)).to(self.device)
                )

            out = self._module(*tensors)
            if self.device.type == "mps":
                torch.mps.synchronize()

            # Collect on the same side of the lock: a D2H copy racing another
            # thread's submission is the same hazard as the H2D copy above.
            if isinstance(out, (tuple, list)):
                values = [t.detach().float().cpu().numpy() for t in out]
            else:
                values = [out.detach().float().cpu().numpy()]

        results = dict(zip(self._graph_output_names, values))
        if output_names:
            return [results[n] for n in output_names if n in results]
        return values


def clear_cache():
    with _LOCK:
        _CACHE.clear()


# Models whose MPS forward has raised. Keyed by model file so that every session
# for that graph degrades together -- the converted module is shared through
# _CACHE, so re-deciding per session instance would let one model keep failing
# on a different thread while another quietly used the GPU.
_DEGRADED: set = set()


def _is_degraded(model_file: str) -> bool:
    with _LOCK:
        return model_file in _DEGRADED


def degraded_models() -> list:
    """Model files currently running on the CPU fallback, for diagnostics."""
    with _LOCK:
        return sorted(_DEGRADED)


# Optional UI callback, set by the UI layer at startup. The backend deliberately
# knows nothing about Qt; it just hands over the model name and the reason.
_FALLBACK_NOTIFIER = None


def set_fallback_notifier(fn) -> None:
    """Register a callable ``fn(model_name, reason)`` shown when a model degrades."""
    global _FALLBACK_NOTIFIER
    _FALLBACK_NOTIFIER = fn