# Agentic Dataset for AIPerf

Preprocessing and benchmark scripts for running [sammshen/lmcache-agentic-traces](https://huggingface.co/datasets/sammshen/lmcache-agentic-traces) with [AIPerf](https://github.com/ai-dynamo/aiperf).

The dataset contains 787 multi-turn agentic LLM sessions (24,881 total turns) from SWE-bench, GAIA, and WildClaw. Each turn has cumulative OpenAI-format messages including tool calls.

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

| HF field | mooncake_trace field | Transform |
|---|---|---|
| `input` | `messages` | Rename (OpenAI-format messages passed as-is) |
| `session_id` | `session_id` | Direct |
| `output_length` | `output_length` | Direct |
| `pre_gap` | `delay` | Seconds to milliseconds (`int(pre_gap * 1000)`) |

The `pre_gap` / `delay` field is the real tool-execution and user-thinking time between the previous response completing (last streamed token) and the current request being sent. This allows trace replay tools to faithfully reproduce the original inter-request timing without baking in the original server's inference latency.

## 2. Run with AIPerf

AIPerf guarantees that requests within the same `session_id` run **strictly sequentially** — request N+1 only fires after request N completes. This is essential for KV cache benchmarking where each request's prefix must be cached before the next request arrives. This ordering is enforced in all scheduling modes.

### Recommended: `--concurrency`

The simplest correct option for KV cache benchmarking. Maintains N active sessions, each with sequential turns. When a session finishes, a new one starts immediately.

```bash
aiperf profile \
    --model <your-model> \
    --url http://localhost:8000 \
    --endpoint-type chat \
    --streaming \
    --input-file /workspace/results/lmcache_traces.jsonl \
    --custom-dataset-type mooncake_trace \
    --concurrency 20 \
    --request-timeout-seconds 3600 \
    --output-artifact-dir /workspace/results/aiperf_artifacts \
    --extra-inputs ignore_eos:true \
    --export-level records \
    --random-seed 42
```

Adjust `--concurrency` to control how many simultaneous users/sessions you want to simulate.

### Alternative: `--user-centric-rate` (realistic steady-state)

For simulating realistic production traffic patterns with controlled QPS:

```bash
aiperf profile \
    --model <your-model> \
    --url http://localhost:8000 \
    --endpoint-type chat \
    --streaming \
    --input-file /workspace/results/lmcache_traces.jsonl \
    --custom-dataset-type mooncake_trace \
    --user-centric-rate 10 \
    --num-users 20 \
    --request-timeout-seconds 3600 \
    --output-artifact-dir /workspace/results/aiperf_artifacts \
    --extra-inputs ignore_eos:true \
    --export-level records \
    --random-seed 42
```

This simulates 20 concurrent users with turns spaced by `num_users / qps = 2s`. New users spawn to replace finished ones, maintaining steady-state throughput.

### Scheduling modes comparison

| Mode | How it works | Best for |
|---|---|---|
| `--concurrency N` | N active sessions, next turn fires immediately on completion | KV cache benchmarking (max cache pressure, simple) |
| `--user-centric-rate Q --num-users N` | N users, turns spaced by `N/Q` seconds, steady-state from t=0 | Realistic production simulation |
| `--request-rate Q` | Open-loop at Q QPS, skips ticks when all sessions blocked | Stress testing (can underperform target if sessions are long) |
| `--fixed-schedule` | Replays at exact timestamps from `delay` field | Reproducing original trace timing |

All modes guarantee sequential intra-session ordering via a credit-return mechanism — request N+1 is only created after request N completes.

### Key flags

| Flag | Description |
|---|---|
| `--concurrency N` | Max concurrent sessions (conversations, not individual requests) |
| `--request-timeout-seconds 3600` | Long timeout needed for multi-turn sessions |
| `--extra-inputs ignore_eos:true` | Generate exactly `output_length` tokens (needed to match original trace) |
| `--random-seed 42` | Deterministic session ordering for reproducibility |
| `--dataset-sampling-strategy random` | Resample sessions with replacement if you need more data than 787 sessions |
