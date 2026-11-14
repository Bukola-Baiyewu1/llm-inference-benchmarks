# Results

## LLM serving: vLLM vs Hugging Face

Model `Qwen/Qwen2.5-0.5B-Instruct`, greedy decoding, 128 generated tokens per request.

| Backend | Batch | Tokens/s | Batch latency (s) |
| --- | --- | --- | --- |
| vllm | 1 | 175 | 0.73 |
| vllm | 4 | 695 | 0.74 |
| vllm | 16 | 2241 | 0.91 |
| vllm | 64 | 5698 | 1.44 |
| vllm | 128 | 9032 | 1.81 |
| hf | 1 | 29 | 4.43 |
| hf | 4 | 125 | 4.10 |
| hf | 16 | 486 | 4.21 |

| Batch | vLLM speed-up over HF | First-32-token match rate |
| --- | --- | --- |
| 1 | 6.1× | 0% |
| 4 | 5.6× | 0% |
| 16 | 4.6× | 0% |

vLLM throughput rose 51× from batch 1 to 128, while batch latency rose from 0.73 s to 1.81 s.

![Throughput vs batch size](throughput.png)

## Environment

GPU: Tesla T4 · PyTorch 2.11.0+cu130 · CUDA 13.0 · commit d049266 · 2026-11-14T09:55:58Z
