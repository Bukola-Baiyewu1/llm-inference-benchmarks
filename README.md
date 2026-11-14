# LLM Inference Benchmarks

Measured, reproducible experiments on what makes model inference fast or slow on a GPU, from
single kernels up to serving a language model. Each experiment is a command that writes a
JSON result (with GPU, library versions and git commit), and one command turns them all into
a report.

| Experiment | Question it answers | Command |
| --- | --- | --- |
| Roofline | Why is a matrix multiply limited by compute, but an add by memory bandwidth? | `python -m infbench.roofline` |
| Profiling | Where does the time go in a training step, and what does a CPU–GPU sync cost? | `python -m infbench.profiling` |
| Triton kernels | Can I write a correct GPU kernel, and does fusing two operations save time? | `python -m infbench.triton_kernels` |
| JAX sharding | When does splitting a matmul across devices require communication? | `python -m infbench.jax_sharding` |
| LLM serving | How much faster is vLLM than plain Hugging Face, and what does batching cost in latency? | `python -m infbench.serving` |

## Results

> Paste `results/summary.md` here after running the experiments (see `GUIDE.md`).

![Throughput vs batch size](results/throughput.png)

**Key findings:** _(3–4 sentences in your own words after your run: the matmul vs add numbers,
the vLLM speed-up and latency trade-off, the fusion speed-up, and where the all-reduce appeared.)_

## How each experiment is made trustworthy

- **Correct before fast.** Triton kernels are checked against PyTorch with `torch.allclose` on
  awkward sizes (1, 1,023, 1,025, 1,000,003) that exercise the masked last block, before any
  timing. vLLM and Hugging Face outputs are compared token by token under greedy decoding.
- **Fair comparisons.** Every serving request generates exactly the same number of tokens
  (end-of-sequence ignored), with the same prompts, model and precision for both backends.
- **Honest timing.** GPU work is asynchronous, so timers synchronise the GPU before reading
  the clock; warm-up runs are excluded; the median of repeated runs is reported.
- **Reproducible.** Every result file records the GPU, PyTorch, CUDA and git commit.
- **Tested.** 9 CPU tests run in CI, including the real Triton kernels in Triton's interpreter
  mode and the JAX sharding check on 8 simulated devices.

## Reproduce

```bash
git clone https://github.com/Bukola-Baiyewu1/llm-inference-benchmarks.git
cd llm-inference-benchmarks
pip install -e ".[dev]"
pytest -q                                   # CPU tests, no GPU needed
python -m infbench.roofline
python -m infbench.profiling
python -m infbench.triton_kernels           # needs a CUDA GPU
python -m infbench.jax_sharding             # CPU is fine (simulates 8 devices)
pip install -e ".[serving]"
python -m infbench.serving --backend vllm --batches 1 4 16 64 128 --out results/serving_vllm.json
python -m infbench.serving --backend hf --batches 1 4 16 --out results/serving_hf.json
python -m infbench.report                   # writes results/summary.md and throughput.png
```

Tested on Google Colab with an NVIDIA T4 (16 GB). The T4 has no bfloat16 support, so the
serving benchmark uses float16 (`--dtype half`).

## Limitations

- One small model (Qwen2.5-0.5B-Instruct) on one GPU type; numbers will differ on other hardware.
- Offline batch serving only: latency is measured per batch, not per streamed token under live traffic.
- The JAX experiment simulates devices on a CPU: it shows *where* communication happens, not how long it takes on real TPUs.
