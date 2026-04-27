"""Shared utilities: device selection, VRAM reporting, parameter counting."""

import os


def get_device(prefer="auto"):
    """Return a torch device string. 'auto' = cuda if available, else cpu."""
    import torch
    if prefer == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(prefer)


def device_summary():
    """One-line description of the active compute device."""
    import torch
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        names = [torch.cuda.get_device_name(i) for i in range(n)]
        free, total = torch.cuda.mem_get_info()
        return f"CUDA ({n} GPU): {names}, {free/1e9:.1f}/{total/1e9:.1f} GB free"
    return "CPU only (no CUDA detected)"


def count_params(module):
    return sum(p.numel() for p in module.parameters())


def vram_estimate_mb(num_params, dtype_bytes=4, optimizer="adam"):
    """Rough VRAM lower bound for params + grads + optimizer state.

    Adam keeps 2 extra fp32 tensors per param. Activations are extra and
    depend on batch size and seq len.
    """
    factor = {"adam": 4, "sgd": 2, "none": 1}.get(optimizer, 4)
    return num_params * dtype_bytes * factor / (1024 ** 2)
