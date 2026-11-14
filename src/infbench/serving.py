"""LLM serving benchmark: vLLM vs plain Hugging Face generation.

Sends the same prompts at increasing batch sizes and records throughput
(generated tokens per second) and batch latency. Every request generates
exactly ``max_tokens`` tokens (EOS is ignored), so the comparison is fair.
Greedy decoding is used, so the two backends' outputs can also be compared
token by token as a correctness check.

Run each backend in its own process (they each want most of the GPU memory):

python -m infbench.serving --backend vllm --batches 1 4 16 64 128 --out results/serving_vllm.json
python -m infbench.serving --backend hf   --batches 1 4 16       --out results/serving_hf.json
python -m infbench.report
"""
from __future__ import annotations

import argparse
import time
from typing import Protocol

from .common import env_info, save_json

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

TOPICS = ["how a GPU works", "the water cycle", "why the sky is blue", "how vaccines work",
          "what a database index is", "how TCP handshakes work", "photosynthesis",
          "how compilers optimise code", "the causes of inflation", "how neural networks learn"]


def make_prompts(n: int) -> list[str]:
    """Deterministic, varied prompts so results are repeatable."""
    return [f"Explain {TOPICS[i % len(TOPICS)]} in simple terms. (request {i})" for i in range(n)]


class Backend(Protocol):
    name: str

    def generate(self, prompts: list[str], max_tokens: int) -> list[list[int]]:
        """Return the generated token ids for each prompt."""


class FakeBackend:
    """A stand-in with predictable timing, used by the tests (no GPU, no model)."""
    name = "fake"

    def __init__(self, seconds_per_step: float = 0.001):
        self.seconds_per_step = seconds_per_step

    def generate(self, prompts, max_tokens):
        time.sleep(self.seconds_per_step * max_tokens)   # one decode step serves the whole batch
        return [[(len(p) + t) % 1000 for t in range(max_tokens)] for p in prompts]


class VLLMBackend:
    name = "vllm"

    def __init__(self, model: str, dtype: str, gpu_memory_utilization: float):
        from vllm import LLM
        self.llm = LLM(model=model, dtype=dtype, gpu_memory_utilization=gpu_memory_utilization, seed=0)

    def generate(self, prompts, max_tokens):
        from vllm import SamplingParams
        params = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
        outs = self.llm.generate(prompts, params, use_tqdm=False)
        return [list(o.outputs[0].token_ids) for o in outs]


class HFBackend:
    name = "hf"

    def __init__(self, model: str, dtype: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model, padding_side="left")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        torch_dtype = {"half": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
        self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch_dtype).to("cuda").eval()

    def generate(self, prompts, max_tokens):
        enc = self.tok(prompts, return_tensors="pt", padding=True).to("cuda")
        with self.torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=max_tokens, min_new_tokens=max_tokens,
                                      do_sample=False, pad_token_id=self.tok.pad_token_id)
        self.torch.cuda.synchronize()
        return out[:, enc["input_ids"].shape[1]:].tolist()


def sweep(backend: Backend, batches: list[int], max_tokens: int, repeats: int = 1) -> dict:
    """Throughput and latency at each batch size, plus the outputs for a correctness check."""
    backend.generate(make_prompts(2), 8)                  # warm-up: compile, allocate caches
    rows, outputs = [], {}
    for b in batches:
        prompts = make_prompts(b)
        best = None
        for _ in range(repeats):
            t0 = time.perf_counter()
            toks = backend.generate(prompts, max_tokens)
            secs = time.perf_counter() - t0
            if best is None or secs < best[0]:
                best = (secs, toks)
        secs, toks = best
        n_tokens = sum(len(t) for t in toks)
        rows.append({"batch": b, "seconds": secs, "generated_tokens": n_tokens,
                     "tokens_per_s": n_tokens / secs, "batch_latency_s": secs,
                     "ms_per_token_per_request": 1000 * secs / max_tokens})
        outputs[str(b)] = toks
        print(f"[{backend.name}] batch {b:4d}: {n_tokens / secs:9.1f} tok/s, latency {secs:6.2f} s")
    return {"backend": backend.name, "max_tokens": max_tokens, "rows": rows, "outputs": outputs}


def token_match_rate(a: list[list[int]], b: list[list[int]], first_k: int = 32) -> float:
    """Share of requests whose first ``first_k`` greedy tokens are identical in both backends.

    Small float differences (fp16 kernels, batching) can flip a near-tie at some
    step, so a rate a little below 100% is normal; a low rate means a real bug.
    """
    pairs = list(zip(a, b))
    if not pairs:
        return 0.0
    return sum(x[:first_k] == y[:first_k] for x, y in pairs) / len(pairs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["vllm", "hf", "fake"], required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--dtype", default="half", help="half on a T4 (no bfloat16 support)")
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 4, 16, 64, 128])
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.backend == "vllm":
        backend = VLLMBackend(a.model, a.dtype, a.gpu_mem)
    elif a.backend == "hf":
        backend = HFBackend(a.model, a.dtype)
    else:
        backend = FakeBackend()
    res = sweep(backend, a.batches, a.max_tokens, a.repeats)
    res.update({"model": a.model, "dtype": a.dtype, "env": env_info()})
    print("saved", save_json(res, a.out))


if __name__ == "__main__":
    main()
