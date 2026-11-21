"""Python 3.14 workaround. Import this before transformers.

torch 2.9 raises an error for any use of torch.compile on Python 3.14+, and
transformers' ModernBERT code applies @torch.compile when the module is
imported. We never use the compiled path (reference_compile=False), so on
3.14+ torch.compile is replaced by a no-op that returns the function unchanged.
"""

import sys

import torch


def _no_compile(model=None, **kwargs):
    if model is None:  # used as @torch.compile(...)
        return lambda fn: fn
    return model       # used as @torch.compile or torch.compile(fn)


def patch_torch_compile(force: bool = False) -> None:
    if force or sys.version_info >= (3, 14):
        torch.compile = _no_compile


patch_torch_compile()
