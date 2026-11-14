# Results

## Compute-bound vs memory-bound

| Operation | Size | Result | Arithmetic intensity (FLOP/byte) |
| --- | --- | --- | --- |
| matmul | 512 | 5.2 TFLOPS | 171 |
| matmul | 1024 | 16.6 TFLOPS | 341 |
| matmul | 2048 | 23.1 TFLOPS | 683 |
| matmul | 4096 | 22.4 TFLOPS | 1.37e+03 |
| add | 50000000 | 247 GB/s | 0.167 |

## Profiling

Top GPU kernels by self time in an MLP training step:

- 28.6% volta_sgemm_128x64_tn
- 26.7% volta_sgemm_128x64_nt
- 23.8% void at::native::(anonymous namespace)::multi_tensor_apply_kernel<at::native::(a
- 19.4% volta_sgemm_128x64_nn
- 0.4% void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::M

Matrix-multiply kernels: 74.7% of measured time.

Calling `.item()` every step changed CPU loop time from 2.6 ms to 11.6 ms over 10 steps.

## Triton kernels

Correct on all 6 test sizes (including partial last blocks): yes.

| Kernel | Time (ms) |
| --- | --- |
| PyTorch add | 2.463 |
| Triton add | 2.502 |
| PyTorch add, then ReLU | 4.140 |
| Triton fused add + ReLU | 2.488 |

Fusion speed-up: 1.66×. Triton add bandwidth: 240 GB/s.

## JAX sharding on a 4 × 2 device mesh

| Sharded dimension | Communication in compiled program | Matches unsharded |
| --- | --- | --- |
| batch | none | True |
| contraction | all-reduce | True |

## LLM serving: vLLM vs Hugging Face

Model `Qwen/Qwen2.5-0.5B-Instruct`, greedy decoding, 128 generated tokens per request.

| Backend | Batch | Tokens/s | Batch latency (s) |
| --- | --- | --- | --- |
| vllm | 1 | 177 | 0.72 |
| vllm | 4 | 697 | 0.73 |
| vllm | 16 | 2259 | 0.91 |
| vllm | 64 | 5763 | 1.42 |
| vllm | 128 | 9226 | 1.78 |
| hf | 1 | 30 | 4.29 |
| hf | 4 | 127 | 4.02 |
| hf | 16 | 504 | 4.06 |

| Batch | vLLM speed-up over HF | First-32-token match rate | Mean matching prefix (tokens) |
| --- | --- | --- | --- |
| 1 | 5.9× | 100% | 128 of 128 |
| 4 | 5.5× | 100% | 117 of 128 |
| 16 | 4.5× | 100% | 124 of 128 |

vLLM throughput rose 52× from batch 1 to 128, while batch latency rose from 0.72 s to 1.78 s.

![Throughput vs batch size](throughput.png)

## Environment

GPU: Tesla T4 · PyTorch 2.11.0+cu130 · CUDA 13.0 · commit 16d9b49 · 2026-11-14T11:48:52Z
