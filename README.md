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

Four-field conversion, no content modification:

| HF field | mooncake_trace field | Transform |
|---|---|---|
| `input` | `messages` | Rename (OpenAI-format messages passed as-is) |
| `session_id` | `session_id` | Direct |
| `session_timestamp` | `timestamp` | Seconds to milliseconds |
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
    --fixed-schedule \
    --fixed-schedule-auto-offset \
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

- `--fixed-schedule` — replays requests at the original timestamps from the trace. Remove this to send requests as fast as possible (open-loop).
- `--fixed-schedule-auto-offset` — normalizes timestamps to start from 0.
- `--concurrency 128` — max concurrent requests in flight.
- `--dataset-sampling-strategy random` — add this if you need more data than the 764 sessions provide (resamples with replacement).
