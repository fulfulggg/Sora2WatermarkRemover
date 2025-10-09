"""
HF Hub compatibility shim for diffusers==0.26.3 on newer huggingface_hub (>=0.34).

- diffusers==0.26.3 expects `huggingface_hub.cached_download`
- newer huggingface_hub removed it in favor of `hf_hub_download`

This shim attaches `cached_download` symbol that delegates to `hf_hub_download`
at interpreter startup (Python auto-imports `sitecustomize` if present).

Notes:
- This affects any Python process started from the repository root (e.g., Colab's
  `!python remwm.py` after `cd /content/Sora2WatermarkRemover`).
- If you run from a different directory, ensure the repo root is on PYTHONPATH
  (e.g., `PYTHONPATH=. python remwm.py` from the repo root).
"""

import importlib
import sys


def _apply_hf_hub_compat():
    try:
        hfh = importlib.import_module("huggingface_hub")
    except Exception:
        # huggingface_hub not installed yet at interpreter init time
        return

    # Only attach if missing and the modern API exists
    has_cached = hasattr(hfh, "cached_download")
    has_new = hasattr(hfh, "hf_hub_download")
    if not has_cached and has_new:
        def cached_download(*args, **kwargs):
            # Keep signature-agnostic; delegate directly
            return hfh.hf_hub_download(*args, **kwargs)

        # Attach to module attribute and sys.modules to satisfy both import patterns:
        # 1) from huggingface_hub import cached_download
        # 2) import huggingface_hub as hfh; hfh.cached_download(...)
        setattr(hfh, "cached_download", cached_download)
        sys.modules["huggingface_hub"].cached_download = cached_download


# Apply on interpreter startup
try:
    _apply_hf_hub_compat()
except Exception:
    # Never block interpreter startup due to the shim
    pass
