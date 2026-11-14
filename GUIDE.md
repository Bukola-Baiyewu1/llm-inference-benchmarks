# Step-by-step guide: run and publish the LLM inference benchmarks

You'll end up with a public repo containing five measured experiments, a results report, a
throughput chart and passing tests. Plan for about 2–3 hours, mostly waiting for Colab.

You already ran most of these ideas in your November labs. This project turns them into
clean, repeatable code with your own fresh numbers. **Use only the numbers from this run in
the README and on your CV.**

---

## Step 1 — Create the repo and upload the code (15 min)

1. github.com → **+** → **New repository**. Name: `llm-inference-benchmarks`. Description:
   `GPU and LLM inference performance: roofline, profiling, Triton kernels, JAX sharding, vLLM vs Hugging Face`.
   **Public**, and leave every "Add …" box **unticked**. Click **Create repository**.
2. Unzip `llm-inference-benchmarks.zip`. On the empty repo page click **uploading an existing file**.
3. Select **everything inside** the unzipped folder (on a Mac, press **Cmd + Shift + .** first to
   show `.github` and `.gitignore`) and drag it in. Commit message: `Initial benchmarks code and tests`.
4. Open the **Actions** tab.

**Success looks like:** the "tests" run turns **green** within about 3 minutes.

**Troubleshooting:** no `.github` folder → hidden files weren't selected, upload again. Red run →
click it, copy the last 20 lines of the failed step, send them to me.

---

## Step 2 — Open Colab with a GPU (2 min)

1. colab.research.google.com → **New notebook**.
2. **Runtime → Change runtime type → T4 GPU → Save**.
3. Run this first cell (Shift + Enter):

```python
!nvidia-smi
!git clone https://github.com/Bukola-Baiyewu1/llm-inference-benchmarks.git
%cd llm-inference-benchmarks
!pip install -q -e ".[dev]"
!pytest -q
```

**Success looks like:** a table showing **Tesla T4**, then `9 passed` (or `8 passed, 1 skipped`).

**Troubleshooting:** `nvidia-smi: command not found` → the GPU runtime isn't on; repeat step 2.2.

---

## Step 3 — The four quick experiments (15 min)

Run each line in its own cell, one at a time, and read the output before moving on.

```python
!python -m infbench.roofline
```
**What it measures:** matrix multiply speed in TFLOPS at four sizes, and add speed in GB/s.
**Success looks like:** matmul TFLOPS rising with size, and one add line in GB/s.

```python
!python -m infbench.profiling
```
**What it measures:** the top 10 operations by GPU time in an MLP training step, and the CPU
loop time with and without `.item()` each step.
**Success looks like:** matrix-multiply operations (names containing `gemm`, `mm` or `addmm`)
at the top, and a higher time with `.item()`.

```python
!python -m infbench.triton_kernels
```
**What it measures:** first, whether your two kernels give the same answer as PyTorch on 6
awkward sizes; then their speed, and the speed-up from fusing add + ReLU.
**Success looks like:** six lines that all say `True`, then the timing lines.

```python
!python -m infbench.jax_sharding
```
**Success looks like:** `batch comm ops: none` and `contraction comm ops: ['all-reduce']`.

---

## Step 4 — The serving benchmark (45–75 min)

1. Install the serving libraries (5–10 minutes):

```python
!pip install -q -e ".[serving]"
```

2. **Runtime → Restart session** (needed after installing vLLM). Then rerun only:

```python
%cd /content/llm-inference-benchmarks
```

3. Run vLLM:

```python
!python -m infbench.serving --backend vllm --batches 1 4 16 64 128 --out results/serving_vllm.json
```

**Success looks like:** five lines like `[vllm] batch 128: … tok/s, latency … s`, with tokens/s
rising a lot as the batch grows.

4. Run Hugging Face in a **separate** command (vLLM must have finished, so the GPU memory is free):

```python
!python -m infbench.serving --backend hf --batches 1 4 16 --out results/serving_hf.json
```

**Success looks like:** three `[hf]` lines, much lower tokens/s than vLLM at the same batch size.

5. Build the report:

```python
!python -m infbench.report
```

**Success looks like:** a printed report ending in `Saved results/summary.md`, with a table that
includes **vLLM speed-up over HF** and a **match rate**.

**About the match rate:** greedy decoding should give the same tokens in both backends, but
float16 kernels can break near-ties differently. A rate around 80–100% is normal. A rate near
0% means something is wrong; send it to me.

**Troubleshooting**
- `CUDA out of memory` in vLLM → add `--gpu-mem 0.7`, or drop `128` from `--batches`.
- vLLM error mentioning **bfloat16** or **compute capability** → check you passed nothing that
  overrides `--dtype half` (the default).
- vLLM install or import error after installing → you skipped the restart; do step 4.2.
- Hugging Face run very slow at batch 16 → that's expected; it's the point of the comparison.

---

## Step 5 — Download and publish the results (20 min)

1. In Colab:

```python
!zip -r results.zip results
from google.colab import files
files.download("results.zip")
```

2. Unzip it. In your GitHub repo open the `results` folder → **Add file → Upload files** → drop
   in all the `.json` files, `summary.md` and `throughput.png`. Commit: `Add T4 results`.
3. Edit `README.md` (pencil icon): replace the grey placeholder under **Results** with the
   contents of `summary.md` (leave out its "# Results" title line), and replace the
   **Key findings** placeholder with 3–4 sentences using your numbers. Commit.

**Success looks like:** the repo front page shows your tables, the throughput chart and your
own findings.

---

## Step 6 — Be ready to explain it (30 min)

Answer out loud, 5–6 sentences each, with **your** numbers:

1. Why does the matmul reach so many more operations per second than the add? (Arithmetic intensity.)
2. What did the profiler show, and why does `.item()` slow the loop down?
3. Why do the Triton kernels need a mask, and why did fusion help?
4. Why did an all-reduce appear only when the contraction dimension was sharded?
5. Why does vLLM get so much faster with larger batches, and why did latency rise?
   (Decoding is memory-bound; one weight load serves many requests; continuous batching and paged attention.)
6. How did you make sure the comparison with Hugging Face was fair, and correct?

---

## Step 7 — Then add it to your CV

Send me `summary.md` and your latest CV. The entry will follow this shape, with your numbers:

> **LLM inference benchmarks (2026)** · github.com/Bukola-Baiyewu1/llm-inference-benchmarks
> Benchmarked GPU and LLM inference on an NVIDIA T4: served Qwen2.5-0.5B with vLLM at [X]
> tokens/s (batch 128), [Y]× faster than Hugging Face at the same batch, with a greedy
> token-match check between them. Wrote and verified Triton kernels (fused add + ReLU [Z]×
> faster than PyTorch), profiled training with torch.profiler, and traced all-reduce
> communication in sharded JAX matmuls. Reproducible results and CI-tested.
