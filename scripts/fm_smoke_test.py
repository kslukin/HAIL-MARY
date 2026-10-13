"""Check that the foundation-model packages import inside the Docker image."""

import importlib

import torch

print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())

for name in ["chronos", "timesfm3", "timesfm"]:
    try:
        importlib.import_module(name)
        print(f"{name}: OK")
    except Exception as exc:  # report every failure instead of stopping at the first
        print(f"{name}: FAILED ({type(exc).__name__}: {exc})")
