"""CPU-only tests. GPU code paths are checked by the scripts themselves (they assert correctness)."""
import json
import os
import subprocess
import sys

import pytest
import torch

from infbench import roofline
from infbench.common import percentiles, time_fn
from infbench.serving import FakeBackend, make_prompts, sweep, token_match_rate


def test_arithmetic_intensity_formulas():
    # 2n^3 FLOPs over 3n^2 numbers of 2 bytes: intensity = n / 3 for fp16
    assert roofline.arithmetic_intensity_matmul(3000, 2) == pytest.approx(1000)
    assert roofline.arithmetic_intensity_add(4) == pytest.approx(1 / 12)
    assert roofline.matmul_flops(10) == 2000
    assert roofline.add_bytes(10, 4) == 120


def test_roofline_measure_runs_on_cpu():
    res = roofline.measure("cpu", torch.float32, [64], 10_000, repeats=2)
    ops = {r["op"] for r in res["rows"]}
    assert ops == {"matmul", "add"}
    assert all(r["seconds"] > 0 for r in res["rows"])


def test_percentiles_and_timer():
    p = percentiles(list(range(1, 101)))
    assert p["p50"] == pytest.approx(50.5) and p["p99"] == pytest.approx(99.01)
    times = time_fn(lambda: sum(range(1000)), warmup=1, repeats=3)
    assert len(times) == 3 and all(t >= 0 for t in times)


def test_prompts_are_deterministic_and_unique():
    assert make_prompts(5) == make_prompts(5)
    assert len(set(make_prompts(50))) == 50


def test_sweep_throughput_scales_with_batch():
    res = sweep(FakeBackend(seconds_per_step=0.0005), [1, 8], max_tokens=20)
    r1, r8 = res["rows"]
    assert r1["generated_tokens"] == 20 and r8["generated_tokens"] == 160
    assert r8["tokens_per_s"] > 4 * r1["tokens_per_s"]   # batching amortises each decode step


def test_token_match_rate():
    a = [[1, 2, 3], [4, 5, 6]]
    assert token_match_rate(a, a) == 1.0
    assert token_match_rate(a, [[1, 2, 3], [4, 5, 7]], first_k=3) == 0.5
    assert token_match_rate(a, [[1, 2, 3], [4, 5, 7]], first_k=2) == 1.0


def test_jax_sharding_all_reduce_only_for_contraction(tmp_path):
    # Fresh process, because JAX's device count is fixed when JAX first starts.
    out = tmp_path / "js.json"
    env = {**os.environ, "XLA_FLAGS": "--xla_force_host_platform_device_count=8",
           "PYTHONPATH": "src", "JAX_PLATFORMS": "cpu"}
    subprocess.run([sys.executable, "-m", "infbench.jax_sharding", "--out", str(out)], check=True, env=env)
    rows = {r["mode"]: r for r in json.loads(out.read_text())["rows"]}
    assert rows["batch"]["has_all_reduce"] is False
    assert rows["contraction"]["has_all_reduce"] is True
    assert rows["batch"]["matches_unsharded"] and rows["contraction"]["matches_unsharded"]


def test_report_builds_summary(tmp_path):
    fake = sweep(FakeBackend(0.0002), [1, 4], max_tokens=8)
    for name in ("serving_vllm.json", "serving_hf.json"):
        (tmp_path / name).write_text(json.dumps({**fake, "model": "fake", "env": {}}))
    subprocess.run([sys.executable, "-m", "infbench.report", "--dir", str(tmp_path)], check=True,
                   env={**os.environ, "PYTHONPATH": "src"})
    text = (tmp_path / "summary.md").read_text()
    assert "vLLM speed-up over HF" in text and "100%" in text
    assert (tmp_path / "throughput.png").exists()


def test_triton_kernels_in_interpreter_mode():
    """Runs the real Triton kernels on CPU through Triton's interpreter (no GPU needed)."""
    pytest.importorskip("triton")
    code = (
        "import torch\n"
        "from infbench.triton_kernels import triton_add\n"
        "for n in (1, 1023, 1025, 5000):\n"
        "    x, y = torch.randn(n), torch.randn(n)\n"
        "    assert torch.allclose(triton_add(x, y), x + y)\n"
        "    assert torch.allclose(triton_add(x, y, relu=True), torch.relu(x + y))\n"
        "print('ok')\n"
    )
    env = {**os.environ, "TRITON_INTERPRET": "1", "PYTHONPATH": "src"}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]


def test_mean_common_prefix():
    from infbench.serving import mean_common_prefix
    assert mean_common_prefix([[1, 2, 3]], [[1, 2, 3]]) == 3
    assert mean_common_prefix([[1, 2, 3], [1, 9, 9]], [[1, 2, 0], [1, 2, 3]]) == 1.5


def test_backends_use_identical_greedy_settings():
    """Guards the bug where HF silently applied the model's repetition penalty."""
    import inspect
    from infbench import serving
    v, h = inspect.getsource(serving.VLLMBackend.generate), inspect.getsource(serving.HFBackend.generate)
    assert "repetition_penalty=1.0" in v and "repetition_penalty=1.0" in h
    assert "min_tokens=max_tokens" in v and "min_new_tokens=max_tokens" in h
