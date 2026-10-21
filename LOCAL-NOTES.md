# Local install notes (Intel Mac / RX 6800 XT)

Deviations from upstream `requirements_mac.txt`, and why.

## torch / torchvision substituted

Upstream pins `torch==2.2.2` / `torchvision==0.17.2` — the last PyPI x86_64 macOS
wheels. Those are CPU-only here and would not use the GPU at all. This install uses
locally built wheels instead:

- `torch 2.14.1` (cp313) — `~/pytorch-build/pytorch/dist/`
- `torchvision 0.29.1a0+1f1a920` (cp313) — `~/pytorch-build/vision-0.29.1/dist/`

These are the same builds ComfyUI runs, including the MPS allocator patch that
removes the `hasUnifiedMemory` gate. Without that patch torch refuses to
initialise MPS on a discrete-GPU Intel Mac.

**Python 3.13 is forced by the dependency set, not a preference.**
`scikit-image==0.26.0` (required by `face_swappers.py` / `faceutil.py` for
`SimilarityTransform.from_estimate`) has no cp310 wheels and requires Python
>=3.11. The available `torch 2.9.1` wheel is cp310-only, so 2.9.1 is unusable
here. 3.13 + torch 2.14.1 is the only combination that satisfies both.

## Only big models belong on MPS

Measured on this machine, same session, ORT CPU vs onnx2torch MPS:

| model | ORT CPU | onnx2torch MPS | |
|---|---|---|---|
| scrfd_2.5g_bnkps | 22.0 ms | 35.5 ms | **0.62x, slower** |
| det_10g | 76.2 ms | 33.2 ms | 2.30x |
| inswapper_128.fp16 | 841.1 ms | 66.4 ms | 12.67x |
| InStyleSwapper256_Version_A.fp16 | 3391.1 ms | 176.0 ms | 19.27x |

Moving a model to MPS is only worth it when its ONNX Runtime CPU time is well
above ~50 ms. Below that the host<->device copy and dispatch cost more than the
arithmetic, and it is a net loss even though the model runs "on the GPU".

This bit: scrfd was on MPS while being the default detector and ~63% of the
per-frame budget, which cancelled out the 12-19x the swappers were getting and
made the whole change look worthless. Removing it took detection from ~380 ms to
~135 ms per call.

Before adding a model to `mps_backend.MPS_MODELS`, benchmark both paths.

## Benchmark with nothing else on the GPU

ComfyUI holds MPS models resident even when idle, and it was running during most
of this work. It inflated every measurement by roughly 2x:

| | ComfyUI running | ComfyUI stopped |
|---|---|---|
| inswapper, standalone | 53 ms | 51 ms |
| inswapper, in-pipeline | 165 ms | 98 ms |
| steady-state frame | ~500 ms | **~280 ms** |

Overall the pipeline went ~930 ms -> ~280 ms per frame (3.3x), output unchanged
(per-frame delta vs input identical to the pre-change baseline). Any future
timing here should stop ComfyUI first, or the numbers are meaningless.

## Unresolved: inswapper is 98 ms in-pipeline vs 51 ms standalone

Called directly in the same process, after the pipeline finishes, it is 51.9 ms.
Every isolated component is also ~51 ms. Ruled out, each by direct measurement:
idle gaps (0), concurrent ORT load (+27 ms), ORT sessions merely open (+3 ms), a
second resident MPS model (0), input dtype/device (+0), the io_binding wrapper
(+0.4 ms), and a full CPU burst of parser+detector+embedder before the call
(+7 ms). The residual ~47 ms is something the pipeline does while running that
was not isolated.

It does not change the decision: ONNX Runtime CPU is 841 ms for this model, so
MPS wins 9-16x either way.

## ort_ready() must not wrap bind_output

`platform_support.ort_ready()` returns a *new* tensor when the input is not on
CPU. Wrapping a `bind_output` argument with it and taking `.data_ptr()` leaves a
dangling pointer, because the temporary dies at the end of the expression. Inputs
are fine (the buffer is read during the call); outputs are written afterwards.
24 bind_output sites were wrapped this way and all had to be reverted. Only
`bind_input` may be wrapped.

## MPS failures fall back to CPU, per model

A converted graph can hit an MPS kernel that this torch build does not carry,
and some only fail for a particular input shape. When that happens
`TorchInferenceSession.run_outputs` catches it, moves **that model file** to
ONNX Runtime on the CPU, and keeps it there for the rest of the run. Degradation
is keyed by model file rather than by session instance because the converted
module is shared through `_CACHE` -- deciding per instance would let one thread
keep retrying a graph another thread had already given up on.

The fallback is sticky on purpose. The failure is a property of the graph, so
retrying every frame would pay the exception cost forever for no gain. Restarting
is what re-probes it.

Verified: forced MPS failure, then confirmed no exception propagated, the CPU
result matched a reference ORT CPU session to max|diff| 0.0000, the model reported
as degraded, five subsequent calls did not re-enter the MPS forward (0 calls),
and the `io_binding` path falls back as well.

**What this does not cover:** a Metal command buffer assertion
(`-[MTLIOAccelCommandBuffer validate] ... failed assertion`) aborts the process
outright and cannot be caught in Python. The fallback only handles exceptions MPS
raises into Python.

## MPS inference must be fully serialised (Metal abort)

The app runs four persistent worker threads, and each of them calls
`session.run_with_iobinding()` concurrently. MPS command buffers are shared per
process, so two threads submitting at the same time abort the whole process:

```
-[MTLIOAccelCommandBuffer validate]:207: failed assertion
`commit an already committed command buffer'
-[MTLIOAccelCommandBuffer validate]:208: failed assertion
`commit command buffer with uncommitted encoder'
```

Exit code 134, no Python traceback, nothing catchable.

Every MPS touch point therefore has to sit inside one backend lock
(`mps_backend._LOCK`) — not just the forward pass:

- the host→device copy into the module inputs
- the forward pass
- `torch.mps.synchronize()` after it
- the device→host collection of results
- `TorchIOBinding.synchronize_outputs()`, which `run_ort_with_iobinding()` calls
  from the worker thread *after* `run_with_iobinding()` returns, so it is not
  covered by the lock in `run_outputs()` and has to take it itself

Locking only the forward pass is not sufficient. Measured with a standalone
reproducer at 8 threads × 40 iterations:

| variant | result |
|---|---|
| transfer outside the lock | SIGABRT |
| transfer + forward + collection locked, `synchronize_outputs` not | 2 of 3 runs SIGABRT |
| all five points locked | 3 of 3 runs clean |

## torch-on-MPS for the pipeline's own math: available, NOT verified

`platform_support.torch_processing_device()` picks the device for the pure-torch
per-frame chain (warps, masks, blending, colour). It is a separate attribute from
`ModelsProcessor.device`, which must stay CPU because it is the tag handed to
`io_binding.bind_input(device_type=...)`; ORT cannot dereference a Metal pointer,
and a wrong buffer there is silent corruption rather than an error.

The speedup is real. Chained with the frame resident on the GPU, one frame's torch
work at 1080p measured 219.8 ms on CPU vs 26.9 ms on MPS (8.15x). Do NOT judge this
from single-op microbenchmarks: interpolate measured 0.19x and sigmoid 0.29x on
MPS, because each paid a host-to-device copy of the whole frame for one op.

**Leave it off. Do not set `VISOMASTER_TORCH_DEVICE=mps`.** Two separate
problems, both local to this machine:

1. *Wrong output.* Swapper and mask output buffers were allocated with
   `torch.empty_like(<frame tensor>)`, so on MPS they inherited the frame's
   device and reached `io_binding` as `buffer_ptr`. The MPS backend fills those
   by rebuilding a numpy view over the pointer, and an MPS tensor's
   `data_ptr()` is the `MTLBuffer` *object* address here, not graphics memory —
   so the write landed on the Objective-C heap. That is the dark face (mean 50
   vs 122 on CPU over the face ROI) and the *first* swap of a run reading all
   zeros. Separately, `bincount`, `cumsum`, `nonzero`, `masked_select`, boolean
   mask indexing and bool `.sum()` have no working Metal kernel at all, which
   aborts colour transfer with "SC compilation failure".

2. *Segfault.* The worker pool runs several FrameWorkers and this torch
   build's MPS backend does not survive that. A thread doing `conv2d` plus an
   `.item()` sync survives alone and dies with SIGSEGV at two or more, with no
   Python traceback — it takes the whole app down mid-render. Locking
   `torch.mps.synchronize` does not help, and holding one lock across the whole
   frame does fix it but then serialises every thread, so it is no faster than
   staying on CPU.

The output bugs are fixed on the `visomaster-mps-wip` branch and are
bit-identical to CPU here, but they are not on `master` because the segfault
makes the path unusable regardless. A silently wrong face — or a dead app — is
worse than no speedup.

Note separately: thread count dominates this workload. At 512x512 the same
frame measured 1519 ms/frame with 4 worker threads, 777 ms with 2 and 488 ms
with 1. Lowering `nThreadsSlider` is a ~3x win with no risk, and unlike the MPS
path it costs nothing in correctness. 4 is the shipped default and suits CUDA
machines; on this one it is a pessimisation.

Two MPS gaps had to be closed first, both the same broken Metal kernel family as
conv3d and SDPA prefill in the local torch build ("SC compilation failure / call
to an undefined label"):

- `torch.linalg.inv` and `torch.linalg.det` both abort on MPS. Kornia's
  `warp_affine` calls the inverse twice, so *every* warp in the app died.
  `faceutil.warp_affine` now does the same transform with a 3x3 adjugate inverse
  instead; verified against Kornia on CPU to 6.4e-05 max on a real frame
  (0.016 of one 8-bit LSB). All 16 `kgm.warp_affine` call sites use it.
- Anything bound into ORT goes through `platform_support.ort_ready()`.

Device-mixing traps hit while wiring this, all the same shape -- pure-torch helper
allocates on `.device` (CPU) but is combined with a tensor that followed the frame
onto MPS: `get_border_mask`, `get_dynamic_side_mask`, the `swap_mask` allocation,
and the affine matrix passed to `warp_affine`. Each now takes its device from the
tensor it is combined with.

## Baseline: MPS is not used for model inference (upstream behaviour)

This is upstream's documented decision (`docs/macos.md`), not a local one. Kept
for reference — the section after it changes this for the face swappers.
Inference binds torch tensors straight into ONNX Runtime via `io_binding`
(~74 call sites). ONNX Runtime has no `mps` device and would dereference a Metal
buffer pointer as garbage rather than failing loudly, so tensors are kept on CPU
and execution runs on the CPU or CoreML provider.

Upgrading torch therefore does not make the face-swap models GPU-accelerated. It
affects the surrounding torch code (pre/post-processing, restorers). Expect the
performance figures in `docs/macos.md` to apply unless the onnx2torch backend
below is active.

## MPS inference via onnx2torch (local change)

The section above described the stock behaviour. It is no longer accurate: this
checkout now runs the face swappers on MPS.

Inference binds torch CPU tensors straight into ONNX Runtime via `io_binding`.
ONNX Runtime has no `mps` device, so upstream keeps everything on the CPU or
CoreML provider. `app/processors/utils/mps_backend.py` works around that by
converting the ONNX graph to a torch module with `onnx2torch` — torch owns the
buffers, so the graph runs natively on MPS. `TorchInferenceSession` mimics the
slice of the ORT API this codebase actually uses (`get_inputs`, `get_outputs`,
`run`, and an `io_binding` supporting `bind_input`/`bind_output` with a raw
pointer), so **no call site changed** — the hook is 12 lines in the
`ModelsProcessor.load_model` session factory.

Measured here (i9-10900 / RX 6800 XT, per swap, warmed):

| model | CPU | CoreML | MPS | speedup vs CPU |
|---|---|---|---|---|
| Inswapper128 | 560 ms | 526 ms | **52 ms** | 10.7× |
| InStyleSwapper256 Version A | 2426 ms | 2322 ms | **143 ms** | 17.0× |
| alphaface_swapper | 2430 ms | 928 ms | **130 ms** | 18.7× |
| ghost_unet_3_block | 309 ms | 657 ms | **161 ms** | 1.9× |

Fidelity is not a concern: Pearson r ≥ 0.99997 against ONNX Runtime for every
converted model (1.0000000 for four of seven).

### Only some models are converted

`MPS_MODELS` is an **allowlist**, because the small embedders go the other way —
CoreML already beats MPS there, and at 112×112 the per-call MPS dispatch
overhead exceeds the arithmetic:

| model | CPU | CoreML | MPS | winner |
|---|---|---|---|---|
| w600k_r50 (arcface embed) | 35 ms | **6 ms** | 23 ms | CoreML 3.8× |
| simswap_arcface_model | 178 ms | **54 ms** | 96 ms | CoreML 1.8× |

An allowlist rather than a blocklist means a newly added model keeps the
upstream path instead of silently landing on whichever backend is slower for it.

### Control

Settings → General → **MPS Swapper Backend**, right under "Providers Priority".
It defaults to on when `torch.backends.mps.is_available()`.

Toggling it stops processing, drops every loaded session and clears the MPS
cache, because the backend is chosen when a session is *constructed* — without
the reload the toggle would look like it worked while swapping still ran on the
old backend.

The choice is saved to `.mps_backend` next to `app/` so it survives a restart.
A launch-time environment variable still wins over the saved preference, so
`VISOMASTER_MPS_BACKEND=off ./Start_mac.sh` is a reliable one-off override.

```bash
VISOMASTER_MPS_BACKEND=off  ./Start_mac.sh    # stock ONNX Runtime behaviour
VISOMASTER_MPS_BACKEND=auto ./Start_mac.sh    # default
```

`auto` needs no configuration: it uses MPS only when `torch.backends.mps.is_available()`.
Conversion failures fall back to ONNX Runtime with a log line rather than failing
the load.

Two traps hit while building this, both recorded in the module docstring:

- ORT's own `copy_outputs_to_cpu()` **segfaults** on `inswapper_128.fp16` when
  driven through the app's pointer-binding path. That is an ORT bug, not the
  shim; the app avoided it only because CoreML/CPU sessions bind differently.
- Verifying the conversion with `U(0,1)` noise instead of realistic activations
  makes `inswapper_128.fp16` look like a gross failure (r = 0.69). Its 512-d
  source input is an ArcFace embedding, near zero-mean with unit-ish scale;
  feeding it uniform noise drives the model into a regime where its 24 internal
  fp16 `Cast` nodes amplify rounding. On real face input it is r = 0.99999914.

Upgrading torch still does not make the *non-converted* models GPU-accelerated —
the restorers, segmenters and embedders above keep running on CPU/CoreML.

## Yolov11 / Yolo26 face detectors (local addition)

Two Ultralytics face detectors were added behind the existing `DetectorModelSelection`
dropdown. They are **detectors only** — nothing about swapping changed, because the
swap path consumes `bbox + kps_5` and is already detector-agnostic.

Ultralytics ships YOLO11/YOLO26 pretrained on COCO, which has no "face" class, so
neither detects faces as shipped. Both entries therefore use third-party WIDERFace
fine-tunes from HuggingFace (AGPL-3.0, fine under this repo's GPL-3.0):
`deepghs/yolo-face` (YOLO11n) and `poptoz/yolo26-hand-pose-face-detection` (YOLO26s).
That second one is a personal account with no release history — it is the only
YOLO26 *face* model that exists today. Swap it out if a better one appears.

**The two exports have genuinely different output layouts**, which is why they share
`_detect_yolo_box_only(..., output_layout=...)` rather than one decoder:

| Model | `output0` | Layout | NMS |
|---|---|---|---|
| YOLO11n | `[1, 5, 8400]` | one-to-many, `(cx,cy,w,h,score)` | needed |
| YOLO26s | `[1, 300, 6]` | one-to-one, `(x1,y1,x2,y2,conf,cls)` | already done |

YOLO26's end-to-end head returns boxes already in xyxy, so the centre→corner
conversion must be skipped, and the shared `_filter_detections_gpu` NMS is a no-op.

**`np.asarray()` on the output *list* adds an axis.** `_run_model_with_lazy_build_check`
returns `io_binding.copy_outputs_to_cpu()`, which is a **list**. `np.squeeze(np.asarray(outs))`
therefore yields `(1, 1, 300, 6)` rather than `(1, 300, 6)`, and the NMS kernel rejects
the 3-D box tensor with `boxes should be a 2d tensor`. Index `[0]` *before* `asarray`.
The existing YOLO11/12 VR180 detectors get away with `np.squeeze` only because their
`[1,5,N]` output happens to be unaffected by the extra leading axis.

Both models emit **no landmarks**, so `_synthesize_kps_from_bboxes` places the ArcFace
template inside each box. That is why the eye landmark lands ~19 px below RetinaFace's
real one, and why Yolov11's swap is fractionally softer than the reference.

Measured here (CPU provider, one 1920x1080 frame, median of 7):

| Detector | ms |
|---|---|
| Yunet | 13.7 |
| SCRFD | 21.6 |
| Yolov11 | 34.3 |
| RetinaFace | 35.9 |
| Yolov8 | 35.9 |
| Yolo26 | 65.2 |

Yolo26 is the *slowest* despite skipping NMS, because the only public fine-tune is the
**s** variant, not nano. Neither model is in `MPS_MODELS`, so both run on the ORT
provider; adding them would need the re-measurement described above.

## Reproducing this environment

```bash
cd /Volumes/Sdiffusion/python/VisoMaster-Fusion
uv venv --python /Users/carakan/.asdf/installs/python/3.13.16/bin/python3.13 .venv
uv pip install --python .venv/bin/python \
    ~/pytorch-build/pytorch/dist/torch-2.14.1-cp313-cp313-macosx_15_0_x86_64.whl \
    ~/pytorch-build/vision-0.29.1/dist/torchvision-0.29.1a0+1f1a920-cp313-cp313-macosx_15_0_x86_64.whl
grep -vE "^(torch|torchvision)==" requirements_mac.txt > /tmp/vm_reqs.txt
uv pip install --python .venv/bin/python -r /tmp/vm_reqs.txt
.venv/bin/python download_models.py
./Start_mac.sh
```