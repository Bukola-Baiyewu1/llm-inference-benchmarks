"""Compute-bound vs memory-bound: matrix multiply vs element-wise add.

A matmul of two n x n matrices does 2n^3 operations on 3n^2 numbers, so its
arithmetic intensity (operations per byte moved) grows with n: it is limited by
the GPU's maths units. An add does 1 operation per 3 numbers moved, so it is
limited by memory bandwidth. This script measures both.

python -m infbench.roofline --out results/roofline.json
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from .common import env_info, save_json, time_fn


def matmul_flops(n: int) -> float:
    return 2.0 * n ** 3


def add_bytes(n_elements: int, bytes_per_el: int) -> float:
    return 3.0 * n_elements * bytes_per_el      # read a, read b, write c


def arithmetic_intensity_matmul(n: int, bytes_per_el: int) -> float:
    return matmul_flops(n) / (3.0 * n * n * bytes_per_el)


def arithmetic_intensity_add(bytes_per_el: int) -> float:
    return 1.0 / (3.0 * bytes_per_el)


def measure(device: str, dtype: torch.dtype, sizes: list[int], add_elements: int, repeats: int) -> dict:
    cuda = device == "cuda"
    bpe = torch.tensor([], dtype=dtype).element_size()
    rows = []
    for n in sizes:
        a = torch.randn(n, n, device=device, dtype=dtype)
        b = torch.randn(n, n, device=device, dtype=dtype)
        t = float(np.median(time_fn(lambda: a @ b, repeats=repeats, cuda=cuda)))
        rows.append({"op": "matmul", "n": n, "seconds": t,
                     "tflops": matmul_flops(n) / t / 1e12,
                     "intensity_flop_per_byte": arithmetic_intensity_matmul(n, bpe)})
    x = torch.randn(add_elements, device=device, dtype=dtype)
    y = torch.randn(add_elements, device=device, dtype=dtype)
    t = float(np.median(time_fn(lambda: x + y, repeats=repeats, cuda=cuda)))
    rows.append({"op": "add", "n": add_elements, "seconds": t,
                 "gb_per_s": add_bytes(add_elements, bpe) / t / 1e9,
                 "intensity_flop_per_byte": arithmetic_intensity_add(bpe)})
    return {"env": env_info(), "device": device, "dtype": str(dtype), "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    ap.add_argument("--add-elements", type=int, default=50_000_000)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--out", default="results/roofline.json")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    if device == "cpu":
        a.sizes, a.add_elements, a.repeats = sorted({min(s, 1024) for s in a.sizes}), 5_000_000, 5
    res = measure(device, dtype, a.sizes, a.add_elements, a.repeats)
    for r in res["rows"]:
        if r["op"] == "matmul":
            print(f"matmul n={r['n']:5d}: {r['tflops']:7.2f} TFLOPS  (intensity {r['intensity_flop_per_byte']:.0f} FLOP/byte)")
        else:
            print(f"add    n={r['n']}: {r['gb_per_s']:7.1f} GB/s    (intensity {r['intensity_flop_per_byte']:.3f} FLOP/byte)")
    print("saved", save_json(res, a.out))


if __name__ == "__main__":
    main()
