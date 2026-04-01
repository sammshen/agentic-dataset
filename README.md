# Agentic Dataset for AIPerf

Preprocessing and benchmark scripts for running [sammshen/lmcache-agentic-traces](https://huggingface.co/datasets/sammshen/lmcache-agentic-traces) with [AIPerf](https://github.com/NVIDIA/aiperf).

The dataset contains 764 multi-turn agentic LLM sessions (24,816 total turns) from SWE-bench, GAIA, and WildClaw. Each turn has cumulative OpenAI-format messages including tool calls.

## Prerequisites

```bash
pip install orjson datasets
```

## 1. Convert the dataset

The script downloads the HuggingFace dataset and converts it to AIPerf's `mooncake_trace` JSONL format:

```bash
python convert_lmcache_to_mooncake.py \
    --output /workspace/results/lmcache_traces.jsonl
```

If you already have the dataset locally:

```bash
python convert_lmcache_to_mooncake.py \
    --input /path/to/train.jsonl \
    --output /workspace/results/lmcache_traces.jsonl
```

### What it does

Three-field conversion, no content modification:

| HF field | mooncake_trace field | Transform |
|---|---|---|
| `input` | `messages` | Rename (OpenAI-format messages passed as-is) |
| `session_id` | `session_id` | Direct |
| `output_length` | `output_length` | Direct |

## 2. Run with AIPerf

```bash
aiperf profile \
    --model nvidia/Llama-3.3-70B-Instruct-FP8 \
    --url http://localhost:8888 \
    --endpoint-type chat \
    --streaming \
    --input-file /workspace/results/lmcache_traces.jsonl \
    --custom-dataset-type mooncake_trace \
    --request-rate 10 \
    --concurrency 128 \
    --benchmark-duration 1800 \
    --benchmark-grace-period 0 \
    --request-timeout-seconds 3600 \
    --output-artifact-dir /workspace/results/aiperf_artifacts \
    --extra-inputs ignore_eos:true \
    --export-level records \
    --ui-type simple \
    --random-seed 42
```

### Key flags

- `--request-rate 10` — target requests per second. Adjust this to control the load. Without it, requests are sent as fast as possible (limited only by `--concurrency`).
- `--concurrency 128` — max concurrent requests in flight.
- `--dataset-sampling-strategy random` — add this if you need more data than the 764 sessions provide (resamples with replacement).
