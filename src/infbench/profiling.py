"""Where does the time go? Profile an MLP, and measure the cost of CPU-GPU syncs.

python -m infbench.profiling --out results/profiling.json
"""
from __future__ import annotations

import argparse
import time

import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile

from .common import env_info, save_json


def make_mlp(d: int = 2048, layers: int = 4) -> nn.Module:
    mods = []
    for _ in range(layers):
        mods += [nn.Linear(d, d), nn.GELU()]
    return nn.Sequential(*mods)


def profile_mlp(device: str, batch: int, d: int, steps: int) -> list[dict]:
    """Top operations by device time (GPU) or CPU time, over a few training steps."""
    model = make_mlp(d).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    x = torch.randn(batch, d, device=device)
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device == "cuda" else [])
    for _ in range(2):                                   # warm-up outside the profiler
        opt.zero_grad(); model(x).pow(2).mean().backward(); opt.step()
    with profile(activities=acts) as prof:
        for _ in range(steps):
            opt.zero_grad(); model(x).pow(2).mean().backward(); opt.step()
        if device == "cuda":
            torch.cuda.synchronize()
    events = prof.key_averages()
    if device == "cuda":
        # Count only real GPU kernels, by their own (self) time. High-level ops such as
        # aten::mm also report the time of the kernels inside them, so summing every
        # event would count the same work twice.
        from torch.autograd import DeviceType
        # record_function labels such as "Optimizer.step#SGD.step" also appear on the GPU
        # timeline but only wrap real kernels, so they are skipped to avoid counting twice.
        items = [(e.key, float(e.self_device_time_total)) for e in events
                 if e.device_type == DeviceType.CUDA and e.self_device_time_total > 0
                 and "#" not in e.key and not e.key.startswith("ProfilerStep")]
    else:
        items = [(e.key, float(e.self_cpu_time_total)) for e in events if e.self_cpu_time_total > 0]
    total = sum(t for _, t in items) or 1.0
    rows = [{"op": k, "time_us": t, "share": t / total} for k, t in sorted(items, key=lambda x: -x[1])]
    return rows


def gemm_share(rows: list[dict]) -> float:
    """Share of measured time spent in matrix-multiply kernels (names with gemm / mm)."""
    keys = ("gemm", "aten::mm", "aten::addmm", "matmul")
    return sum(r["share"] for r in rows if any(k in r["op"].lower() for k in keys))


def sync_cost(device: str, batch: int, d: int, steps: int) -> dict:
    """CPU time for a loop with vs without .item() each step.

    Without .item() the CPU just queues GPU work. With .item() it must wait for
    the GPU to finish every step, so the CPU and GPU can no longer overlap.
    """
    model = make_mlp(d).to(device)
    x = torch.randn(batch, d, device=device)

    def loop(sync_each_step: bool) -> float:
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(steps):
            out = model(x).sum()
            if sync_each_step:
                out.item()
        cpu_ms = (time.perf_counter() - t0) * 1000
        if device == "cuda":
            torch.cuda.synchronize()
        return cpu_ms

    with torch.no_grad():
        loop(False)
        return {"cpu_ms_no_sync": loop(False), "cpu_ms_item_each_step": loop(True), "steps": steps}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--d", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--out", default="results/profiling.json")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = profile_mlp(device, a.batch, a.d, a.steps)
    top, gemm = rows[:10], gemm_share(rows)
    sync = sync_cost(device, a.batch, a.d, a.steps)
    print(f"Top {'GPU kernels' if device == 'cuda' else 'CPU ops'} by self time:")
    for r in top:
        print(f"  {r['share']:6.1%}  {r['op'][:90]}")
    print(f"Matrix-multiply share of {'GPU' if device == 'cuda' else 'CPU'} time: {gemm:.1%}")
    print(f".item() each step: CPU time {sync['cpu_ms_no_sync']:.1f} ms -> {sync['cpu_ms_item_each_step']:.1f} ms")
    print("saved", save_json({"env": env_info(), "device": device, "top_ops": top,
                                  "gemm_share": gemm, "sync": sync}, a.out))


if __name__ == "__main__":
    main()
