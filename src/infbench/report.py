"""Turn the JSON results into one summary (results/summary.md) and a throughput plot.

python -m infbench.report
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .serving import token_match_rate


def load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def serving_section(vllm: dict | None, hf: dict | None) -> list[str]:
    if not vllm and not hf:
        return []
    lines = ["## LLM serving: vLLM vs Hugging Face", ""]
    model = (vllm or hf).get("model", "?")
    lines += [f"Model `{model}`, greedy decoding, {(vllm or hf)['max_tokens']} generated tokens per request.", "",
              "| Backend | Batch | Tokens/s | Batch latency (s) |", "| --- | --- | --- | --- |"]
    for res in (vllm, hf):
        if res:
            for r in res["rows"]:
                lines.append(f"| {res['backend']} | {r['batch']} | {r['tokens_per_s']:.0f} | {r['batch_latency_s']:.2f} |")
    if vllm and hf:
        v = {r["batch"]: r for r in vllm["rows"]}
        h = {r["batch"]: r for r in hf["rows"]}
        lines += ["", "| Batch | vLLM speed-up over HF | First-32-token match rate |", "| --- | --- | --- |"]
        for b in sorted(set(v) & set(h)):
            speed = v[b]["tokens_per_s"] / h[b]["tokens_per_s"]
            match = token_match_rate(vllm["outputs"][str(b)], hf["outputs"][str(b)])
            lines.append(f"| {b} | {speed:.1f}× | {match:.0%} |")
    if vllm:
        first, last = vllm["rows"][0], vllm["rows"][-1]
        lines += ["", f"vLLM throughput rose {last['tokens_per_s'] / first['tokens_per_s']:.0f}× from batch "
                      f"{first['batch']} to {last['batch']}, while batch latency rose from "
                      f"{first['batch_latency_s']:.2f} s to {last['batch_latency_s']:.2f} s."]
    return lines + [""]


def plot_serving(vllm: dict | None, hf: dict | None, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    for res, marker in ((vllm, "o"), (hf, "s")):
        if res:
            ax.plot([r["batch"] for r in res["rows"]], [r["tokens_per_s"] for r in res["rows"]],
                    marker=marker, label=res["backend"])
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("concurrent requests (batch size)"); ax.set_ylabel("generated tokens / s")
    ax.set_title("Throughput vs batch size", fontsize=10); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results")
    a = ap.parse_args()
    d = Path(a.dir)
    lines = ["# Results", ""]
    env = None

    rf = load(d / "roofline.json")
    if rf:
        env = rf["env"]
        lines += ["## Compute-bound vs memory-bound", "", "| Operation | Size | Result | Arithmetic intensity (FLOP/byte) |",
                  "| --- | --- | --- | --- |"]
        for r in rf["rows"]:
            val = f"{r['tflops']:.1f} TFLOPS" if r["op"] == "matmul" else f"{r['gb_per_s']:.0f} GB/s"
            lines.append(f"| {r['op']} | {r['n']} | {val} | {r['intensity_flop_per_byte']:.3g} |")
        lines.append("")

    pf = load(d / "profiling.json")
    if pf:
        env = env or pf["env"]
        s = pf["sync"]
        lines += ["## Profiling", "", "Top operations by time in an MLP training step:", ""]
        lines += [f"- {r['share']:.1%} {r['op']}" for r in pf["top_ops"][:5]]
        lines += ["", f"Calling `.item()` every step changed CPU loop time from {s['cpu_ms_no_sync']:.1f} ms "
                      f"to {s['cpu_ms_item_each_step']:.1f} ms over {s['steps']} steps.", ""]

    tr = load(d / "triton.json")
    if tr:
        b = tr["benchmark"]
        ok = all(c["add_matches_torch"] and c["fused_relu_matches_torch"] for c in tr["checks"])
        lines += ["## Triton kernels", "",
                  f"Correct on all {len(tr['checks'])} test sizes (including partial last blocks): {'yes' if ok else 'NO'}.", "",
                  "| Kernel | Time (ms) |", "| --- | --- |",
                  f"| PyTorch add | {b['torch_add_ms']:.3f} |", f"| Triton add | {b['triton_add_ms']:.3f} |",
                  f"| PyTorch add, then ReLU | {b['torch_add_then_relu_ms']:.3f} |",
                  f"| Triton fused add + ReLU | {b['triton_fused_add_relu_ms']:.3f} |", "",
                  f"Fusion speed-up: {b['fusion_speedup']:.2f}×. Triton add bandwidth: {b['triton_add_gb_per_s']:.0f} GB/s.", ""]

    js = load(d / "jax_sharding.json")
    if js:
        lines += ["## JAX sharding on a 4 × 2 device mesh", "", "| Sharded dimension | Communication in compiled program | Matches unsharded |",
                  "| --- | --- | --- |"]
        for r in js["rows"]:
            lines.append(f"| {r['mode']} | {', '.join(r['communication_ops']) or 'none'} | {r['matches_unsharded']} |")
        lines.append("")

    vllm, hf = load(d / "serving_vllm.json"), load(d / "serving_hf.json")
    lines += serving_section(vllm, hf)
    if vllm or hf:
        plot_serving(vllm, hf, d / "throughput.png")
        lines += ["![Throughput vs batch size](throughput.png)", ""]
        env = env or (vllm or hf)["env"]

    if env:
        lines += ["## Environment", "", f"GPU: {env.get('gpu')} · PyTorch {env.get('torch')} · CUDA {env.get('cuda')} · "
                                        f"commit {env.get('git_commit')} · {env.get('timestamp')}"]
    out = d / "summary.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
