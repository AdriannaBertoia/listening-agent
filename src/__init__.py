# Listening Agent - ADHD Second Brain Helper
# Captures meeting audio, transcribes, and synthesizes action items to Obsidian

# ============================================================================
# PyTorch compatibility patch — MUST run before any model loading imports.
# PyTorch 2.6+ defaults torch.load to weights_only=True, but pyannote/speechbrain
# checkpoints contain serialized omegaconf objects. Lightning Fabric explicitly
# passes weights_only=True, so we force it to False for all loads.
# These are trusted HuggingFace models — this is safe.
# ============================================================================
import torch as _torch
import torch.serialization as _torch_serial

_orig_torch_load = getattr(_torch.load, "__wrapped__", _torch.load)


def _patched_torch_load(*args, **kwargs):
    # Force weights_only=False — lightning/pyannote explicitly pass True
    # but the checkpoints aren't compatible with it
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)


_patched_torch_load.__wrapped__ = _orig_torch_load
_torch.load = _patched_torch_load

# Some libraries reference torch.serialization.load directly
if hasattr(_torch_serial, "load"):
    _torch_serial.load = _patched_torch_load
