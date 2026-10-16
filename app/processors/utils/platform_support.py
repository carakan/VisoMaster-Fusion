"""Execution-provider capability detection.

VisoMaster Fusion targets Windows/Linux with NVIDIA hardware, where the
TensorRT and CUDA execution providers are always assumed present. macOS has
neither, so every code path that hard-codes ``"cuda"`` needs a way to ask what
this machine can actually do.

This module is the single source of truth for that question. It reports the
execution providers that are genuinely usable here (in priority order) and maps
each one to the torch device the rest of the pipeline should run tensors on.

Provider names are the user-facing strings stored in
``control["ProvidersPrioritySelection"]`` and matched in
``ModelsProcessor.update_provider_configuration``.
"""

import os
import platform
import sys
import threading

import numpy as np
import torch

SYSTEM_PLATFORM = platform.system()
IS_MACOS = sys.platform == "darwin"


def _onnxruntime_providers() -> list:
    """Available ONNX Runtime EPs, or an empty list if ORT cannot be queried."""
    try:
        import onnxruntime

        return list(onnxruntime.get_available_providers())
    except Exception:
        return []


def has_cuda() -> bool:
    """True when a usable CUDA device is present."""
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


def has_mps() -> bool:
    """True when torch can run on Apple's Metal Performance Shaders backend."""
    try:
        return torch.backends.mps.is_available()
    except Exception:
        return False


def has_tensorrt() -> bool:
    """True when the TensorRT python bindings import."""
    try:
        import tensorrt  # noqa: F401

        return True
    except Exception:
        return False


def has_coreml() -> bool:
    """True when ONNX Runtime exposes the CoreML execution provider."""
    return "CoreMLExecutionProvider" in _onnxruntime_providers()


def has_neural_engine() -> bool:
    """True on Apple Silicon, which is the only Mac hardware with a Neural Engine.

    Intel Macs expose the CoreML EP too, but it can only fall back to their CPU
    and (via MPSGraph) an integrated/AMD GPU that CoreML drives poorly.
    """
    return IS_MACOS and platform.machine() == "arm64"


def has_mps() -> bool:
    """Whether torch can actually run on MPS.

    This is a real check rather than a platform guess: it imports the optional
    onnx2torch/torch backend and asks the driver, so a mac without a usable
    Metal device never advertises the provider.
    """
    try:
        from app.processors.utils import mps_backend

        return mps_backend.mps_available()
    except Exception:
        return False


def available_execution_providers() -> list:
    """Execution providers usable on this machine, best first.

    Always ends with ``"CPU"``, which is available everywhere and is the
    guaranteed-working fallback.

    ``"MPS"`` leads on macOS when torch can reach Metal. It is not an ONNX
    Runtime provider at all: the models listed in ``mps_backend.MPS_MODELS`` are
    executed by the onnx2torch backend instead, which owns its own MPS device.
    Selecting it is still the right knob, because it is what decides whether
    those models run on the GPU at all.

    On Intel Macs CoreML is offered but deliberately ranked *below* CPU. Measured
    on an i7-9750H / Radeon Pro 5300M across the per-frame hot path
    (det_10g 62ms vs 65ms, w600k_r50 84ms vs 89ms, inswapper_128 1021ms vs
    1244ms), CoreML lost every time: the models carry dynamic dimensions, and
    RequireStaticInputShapes — which is mandatory for correct output, see
    ``ModelsProcessor.coreml_ep_options`` — hands those subgraphs back to the CPU
    EP anyway, leaving only partitioning overhead. Apple Silicon has the Neural
    Engine and a very different profile, so CoreML leads there.
    """
    providers = []
    if has_mps():
        providers.append("MPS")
    if has_cuda():
        if has_tensorrt():
            providers.extend(["TensorRT", "TensorRT-Engine"])
        providers.append("CUDA")
    if has_coreml() and has_neural_engine():
        providers.append("CoreML")
    providers.append("CPU")
    if has_coreml() and not has_neural_engine():
        providers.append("CoreML")
    return providers


def default_execution_provider() -> str:
    """The provider to select when the user has not chosen one."""
    return available_execution_providers()[0]


def onnxruntime_fallback_provider() -> str:
    """Best ONNX Runtime provider for the models that stay on ORT.

    Needed when the chosen provider is ``"MPS"``: MPS is not an ORT provider, so
    the models the onnx2torch backend does not cover still need a real EP list.
    """
    for provider in available_execution_providers():
        if provider != "MPS":
            return provider
    return "CPU"


def torch_device_for_provider(provider_name: str, gpu_id: int = 0) -> tuple:
    """Map a provider name to ``(device, device_type)`` for torch tensors.

    ``device`` is the string handed to ``torch.Tensor.to()``; ``device_type`` is
    the bare backend name used for ``torch.autocast`` and for the ``!= "cpu"``
    checks scattered through the pipeline.
    """
    if provider_name in ("TensorRT", "TensorRT-Engine", "CUDA"):
        return f"cuda:{gpu_id}", "cuda"
    if provider_name in ("CoreML", "MPS"):
        # Torch stays on CPU for both, and for the same reason. The inference
        # path binds torch tensors straight into ONNX Runtime via io_binding
        # (device_type=..., buffer_ptr=tensor.data_ptr()). ORT has no "mps"
        # device and cannot dereference a Metal buffer pointer, so an MPS tensor
        # would be bound as garbage. CoreML still runs the model itself on the
        # GPU/Neural Engine, and MPS runs it through the onnx2torch backend
        # which allocates its own tensors -- only the pre/post-processing and
        # the models still on ORT are CPU-side. The model execution moves to the
        # GPU; the torch tensors do not have to.
        return "cpu", "cpu"
    return "cpu", "cpu"


def default_torch_device() -> str:
    """Torch device matching this machine's default execution provider.

    Deliberately derived from the provider rather than from raw hardware
    capability, so torch tensors always live somewhere the active ONNX Runtime
    provider can bind them.
    """
    return torch_device_for_provider(default_execution_provider())[0]


_TORCH_DEVICE_ENV = "VISOMASTER_TORCH_DEVICE"


def torch_processing_device() -> str:
    """Device for the pipeline's own torch work: warps, masks, blending, colour.

    NOT ``torch_device_for_provider``. Those tensors are pure torch and never
    reach ONNX Runtime, so unlike the ones bound through ``io_binding`` they are
    free to live on the GPU.

    Returns ``"cpu"`` unless ``VISOMASTER_TORCH_DEVICE=mps`` is set explicitly.
    That is deliberate, not an oversight. Chaining a frame's torch work on MPS
    measured 219.8 ms -> 26.9 ms at 1080p on this machine, so the speedup is
    real, but the output has not yet been shown equivalent: with the frame on
    MPS the swapped face comes back dark (mean 50 vs 122 on CPU in the face ROI)
    and the first swap of a run reads all zeros. A silently wrong face is worse
    than no speedup, so this stays opt-in until the numbers match.

    The per-op microbenchmarks are misleading and do not argue either way --
    interpolate measured 0.19x and sigmoid 0.29x on MPS, i.e. slower, because
    each paid a host-to-device copy of the whole frame for a single op. Chained,
    with the frame resident on the GPU, the transfer is paid once per frame.

    Two MPS gaps had to be closed for this path to run at all, both from the
    same broken Metal kernel family as conv3d and SDPA prefill in the local
    torch 2.14.1 build ("SC compilation failure / call to an undefined label"):
    ``torch.linalg.inv`` and ``torch.linalg.det``. See ``faceutil.warp_affine``.
    """
    forced = os.environ.get(_TORCH_DEVICE_ENV, "").strip().lower()
    if forced in ("cpu", "mps"):
        return forced
    return "cpu"


def ort_ready(tensor):
    """A buffer safe to hand to ONNX Runtime through ``io_binding``.

    ``bind_input`` is given a raw ``buffer_ptr`` and no way to check it, so a
    Metal pointer reaching it is silent corruption rather than an error. Call
    this on anything derived from the pipeline frame before it is bound.

    Accepts numpy arrays too -- those are already CPU, but they also expose
    ``data_ptr()`` and so show up at the same binding sites.

    On a torch tensor that is already on CPU this is a contiguity no-op, which
    ORT needs regardless.
    """
    if not hasattr(tensor, "device"):  # numpy array: already host memory
        return np.ascontiguousarray(tensor)
    if tensor.device.type == "cpu":
        return tensor if tensor.is_contiguous() else tensor.contiguous()
    return tensor.detach().cpu().contiguous()


def supports_fp16(device_type: str) -> bool:
    """Whether half precision is worth using on this backend.

    CUDA has fast fp16 throughout. MPS supports fp16 but several ops used by the
    swapper pipeline fall back to (or outright fail on) fp32, so fp16 is only
    enabled for CUDA.
    """
    return device_type == "cuda"


# --- CUDA host-sync policy ---------------------------------------------------
#
# ``cudaStreamSynchronize`` spin-waits under CUDA's default
# ``cudaDeviceScheduleAuto`` policy: a thread waiting on the GPU polls a flag
# rather than sleeping, so every worker blocked on inference burns a core. With
# N worker threads each performing dozens of syncs per frame, that is a large
# amount of CPU spent doing nothing.
#
# An event created with ``cudaEventBlockingSync`` — which is what
# ``torch.cuda.Event(blocking=True)`` gives us — waits on an OS primitive
# instead. Recording it on a stream and synchronizing the event is equivalent to
# synchronizing the stream, but the thread sleeps. Measured on torch 2.11+cu130:
# with work queued, ~23us spinning vs ~39us blocking, so the trade is roughly
# +16us of wake-up latency per sync in exchange for the core.
#
# Set VISOMASTER_CUDA_SPIN_SYNC=1 to restore the old spin-wait behaviour, so the
# two policies can be A/B'd on the same build without a rebuild.

_SPIN_SYNC: bool = os.environ.get("VISOMASTER_CUDA_SPIN_SYNC", "0") == "1"
_TLS = threading.local()
_CUDA_OK: bool | None = None


def _cuda_ok() -> bool:
    """Cached ``has_cuda()``. Called on the per-inference hot path."""
    global _CUDA_OK
    if _CUDA_OK is None:
        _CUDA_OK = has_cuda()
    return _CUDA_OK


def spin_sync_enabled() -> bool:
    """True when the legacy spin-wait sync policy has been forced on."""
    return _SPIN_SYNC


def blocking_stream_sync(stream: "torch.cuda.Stream | None" = None) -> None:
    """Host-wait for a CUDA stream without spinning a core.

    Equivalent to ``stream.synchronize()`` — or
    ``torch.cuda.current_stream().synchronize()`` when *stream* is ``None`` —
    except that the calling thread sleeps instead of polling.

    No-op when CUDA is unavailable, so call sites do not need to guard.
    """
    if not _cuda_ok():
        return

    if _SPIN_SYNC:
        if stream is None:
            torch.cuda.current_stream().synchronize()
        else:
            stream.synchronize()
        return

    # The event is cached per thread: re-recording resets it, so one object per
    # worker serves every sync that worker performs and we avoid allocating a
    # CUDA event on the hot path.
    ev = getattr(_TLS, "sync_event", None)
    if ev is None:
        ev = torch.cuda.Event(blocking=True)
        _TLS.sync_event = ev
    ev.record(stream)
    ev.synchronize()


def sync_device(device_type: str) -> None:
    """Block until queued work on ``device_type`` has completed.

    Replaces bare ``torch.cuda.current_stream().synchronize()`` calls, which
    raise on machines without CUDA. Uses the non-spinning wait above on CUDA.
    """
    if device_type == "cuda" and has_cuda():
        blocking_stream_sync()
    elif device_type == "mps" and has_mps():
        torch.mps.synchronize()


def empty_cache(device_type: str = "") -> None:
    """Release cached allocator memory on whichever backend is active."""
    if has_cuda():
        torch.cuda.empty_cache()
    elif has_mps():
        torch.mps.empty_cache()
