"""Two Triton kernels: a masked vector add, and a fused add + ReLU.

Fusion matters for memory-bound work: unfused, PyTorch reads and writes the
intermediate sum to GPU memory, then reads it again for ReLU. The fused kernel
does both in one pass, so it moves less data.

python -m infbench.triton_kernels --out results/triton.json     (needs a CUDA GPU)
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from .common import env_info, save_json, time_fn

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:  # pragma: no cover
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def _add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr, RELU: tl.constexpr):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n                  # the last block may run past the end of the array
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        out = x + y
        if RELU:
            out = tl.maximum(out, 0.0)
        tl.store(out_ptr + offsets, out, mask=mask)


def triton_add(x: torch.Tensor, y: torch.Tensor, relu: bool = False, block: int = 1024) -> torch.Tensor:
    # CPU tensors are allowed only in Triton interpreter mode (TRITON_INTERPRET=1), used by the tests.
    assert x.shape == y.shape and x.is_contiguous() and y.is_contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    grid = (triton.cdiv(n, block),)
    _add_kernel[grid](x, y, out, n, BLOCK=block, RELU=relu)
    return out


def check_correctness(sizes=(1, 1000, 1023, 1025, 98_432, 1_000_003)) -> list[dict]:
    """Odd sizes on purpose: they test the mask on the last, partial block."""
    rows = []
    for n in sizes:
        x = torch.randn(n, device="cuda"); y = torch.randn(n, device="cuda")
        ok_add = torch.allclose(triton_add(x, y), x + y)
        ok_relu = torch.allclose(triton_add(x, y, relu=True), torch.relu(x + y))
        rows.append({"n": n, "add_matches_torch": bool(ok_add), "fused_relu_matches_torch": bool(ok_relu)})
    return rows


def benchmark(n: int = 50_000_000) -> dict:
    x = torch.randn(n, device="cuda"); y = torch.randn(n, device="cuda")
    t = lambda f: float(np.median(time_fn(f, repeats=20, cuda=True)))
    gbps = lambda bytes_moved, s: bytes_moved / s / 1e9
    bpe = x.element_size()
    res = {
        "torch_add_ms": t(lambda: x + y) * 1e3,
        "triton_add_ms": t(lambda: triton_add(x, y)) * 1e3,
        "torch_add_then_relu_ms": t(lambda: torch.relu(x + y)) * 1e3,
        "triton_fused_add_relu_ms": t(lambda: triton_add(x, y, relu=True)) * 1e3,
    }
    res["triton_add_gb_per_s"] = gbps(3 * n * bpe, res["triton_add_ms"] / 1e3)
    res["fusion_speedup"] = res["torch_add_then_relu_ms"] / res["triton_fused_add_relu_ms"]
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/triton.json")
    a = ap.parse_args()
    if not (HAS_TRITON and torch.cuda.is_available()):
        raise SystemExit("Triton kernels need a CUDA GPU (in Colab: Runtime -> Change runtime type -> T4 GPU).")
    checks = check_correctness()
    for c in checks:
        print(c)
    assert all(c["add_matches_torch"] and c["fused_relu_matches_torch"] for c in checks), "kernel mismatch"
    bench = benchmark()
    for k, v in bench.items():
        print(f"{k}: {v:.3f}")
    print("saved", save_json({"env": env_info(), "checks": checks, "benchmark": bench}, a.out))


if __name__ == "__main__":
    main()
