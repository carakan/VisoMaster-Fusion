"""Smoke test: prove the ONNX Runtime + torch stack actually runs inference."""
import time, numpy as np, onnxruntime as ort, pathlib

ROOT = pathlib.Path("/Volumes/Sdiffusion/python/VisoMaster-Fusion/model_assets")

# A face-embedding model and the actual face-swap model - one of each type.
TESTS = [
    ("ArcFace embed (w600k_r50)", ROOT / "w600k_r50.onnx", (1, 3, 112, 112)),
    ("Face swap (simswap)",        ROOT / "simswap_arcface_model.onnx", (1, 3, 112, 112)),
    ("Face parse (segmentation)",  ROOT / "faceparser_resnet34.onnx", (1, 3, 512, 512)),
]

for provider in ("CPUExecutionProvider", "CoreMLExecutionProvider"):
    print(f"\n=== provider: {provider} ===")
    for label, path, shape in TESTS:
        if not path.is_file():
            print(f"  MISSING {path.name}")
            continue
        try:
            so = ort.SessionOptions()
            so.log_severity_level = 3
            t0 = time.time()
            sess = ort.InferenceSession(str(path), so, providers=[provider])
            load = time.time() - t0
            inp = sess.get_inputs()[0]
            x = np.random.rand(*shape).astype(np.float32)
            t0 = time.time()
            out = sess.run(None, {inp.name: x})
            infer = time.time() - t0
            shapes = [tuple(o.shape) for o in out]
            print(f"  OK  {label:28} load {load:5.2f}s  infer {infer:6.3f}s  out {shapes}")
        except Exception as e:
            print(f"  FAIL {label:28} {type(e).__name__}: {str(e)[:70]}")