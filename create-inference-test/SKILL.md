---
name: create-inference-test
description: Create a post-training inference benchmark script for a model in this repo (clrnet, segformer, rfdetr). Produces a Hydra-configured, pyproject-registered script that runs a correctness parity check against the training-time model, then latency/throughput/memory benchmarks across batch size and input size, logging everything to a dedicated wandb run. Use when the user asks for "inference test", "inference benchmark", "post-training benchmark script", "parity + latency test", or "cable-seg-<model>-inference-test".
---

# Create Inference Test

Generate a dedicated post-training inference benchmark script — a separate job from training, with its own wandb run.

## Step 1 — Confirm the model

Which model is this test for? If the user didn't say, ask:

> "Which model should I create an inference test for? (e.g., `clrnet`, `segformer`, `rfdetr`)"

Use `<MODEL>` for the chosen name. Open the model's training script first to copy its exact checkpoint loader, preprocessing, and wandb project — the parity check only works if the benchmark forward pass matches training.

| Model | Training script | Wandb project |
|---|---|---|
| clrnet | `src/cable_segmentation/clrnet_train.py` | `cable-clrnet` (hardcoded in `wandb.init`) |
| rfdetr | `src/cable_segmentation/rfdetr_train.py` | `${oc.env:WANDB_PROJECT,cable-DF-DETR}` |
| segformer | `src/cable_segmentation/models/segformer/tools/train.py` | read from that script |

## Step 2 — Files to create

1. **Script**: `src/cable_segmentation/<MODEL>_inference_test.py`
2. **Hydra config**: `src/cable_segmentation/configs/<MODEL>_inference_test.yaml`
3. **Entrypoint** — add to `[project.scripts]` in `pyproject.toml`:
   ```toml
   cable-seg-<MODEL>-inference-test = "cable_segmentation.<MODEL>_inference_test:main"
   ```

Invocation contract (this is the required UX):
```bash
cable-seg-<MODEL>-inference-test checkpoint=path/to/checkpoint.ckpt
```

Note: repo convention is `cable-seg-*` prefix, not `seg-cable-*`. Use `cable-seg-<MODEL>-inference-test`.

## Step 3 — Hydra config

Mandatory `checkpoint: ???` so Hydra fails loudly when the user forgets it. See [REFERENCE.md § Config](REFERENCE.md#config-template) for the full YAML; shape summary:

- `checkpoint` (required), `device`, `dtype`
- `parity`: `num_samples`, `atol`, `rtol`, `seed`
- `benchmark`: `batch_sizes`, `input_sizes`, `concurrency_levels`, `warmup_iters` (≥20), `measure_iters` (≥500)
- `wandb`: `project` (from training script), `job_type: inference-benchmark`, `tags: [inference-test]`

## Step 4 — Script responsibilities (in order)

1. `_load_dotenv_for_wandb()` — same helper pattern as `clrnet_train.py` / `rfdetr_train.py`.
2. Load checkpoint using the **exact loader from the training script** — do not invent a new one.
3. `wandb.init(..., job_type="inference-benchmark")` — fresh run, never resume a training run.
4. Log env context: torch/CUDA/driver versions, GPU name, power cap, and attempt `nvidia-smi -lgc` (best effort, non-fatal); log whether clock-locking succeeded.
5. **Correctness parity first.** Seeded fixed inputs → compare this script's forward pass to a reference forward (use the training-time eval path as reference). Compute max abs diff, max rel diff, mismatch fraction. Log `parity/passed`. If it fails, log the failure and **abort before benchmarking** — bad numbers on a broken model are misleading.
6. Warmup: run `warmup_iters` iterations per cell and discard them.
7. Latency matrix: nested loop over `batch_sizes × input_sizes`. Time each iteration with CUDA events (`torch.cuda.Event(enable_timing=True)`), not `time.time()`. Collect ≥`measure_iters` samples. Log p50/p90/p99 + mean + stddev as a wandb `Table` plus individual scalars per cell for easy charting.
8. Throughput under load: sweep `concurrency_levels` with a `ThreadPoolExecutor` submitting forward-pass tasks. Record sustained requests/sec and per-request p50/p99. Log a wandb line plot of latency vs throughput to find the saturation knee.
9. Memory: `torch.cuda.max_memory_allocated()` for peak, rolling window median for steady-state. Log both per cell.
10. `finally:` unlock clocks (`nvidia-smi -rgc`), `wandb.finish()`.

Vision-only scope: prefill/decode, TTFT, ITL, KV-cache do **not** apply to the models in this repo. Do not add those sections. If the user later wants an autoregressive model, see [REFERENCE.md § Autoregressive extensions](REFERENCE.md#autoregressive-extensions).

## Step 5 — Register & smoke-test

1. Add the entrypoint to `pyproject.toml` (keep the list sorted if it already is).
2. `pip3 install -e .` so the new script is on PATH.
3. Smoke test with tiny values before a full run:
   ```bash
   cable-seg-<MODEL>-inference-test checkpoint=<real-ckpt> \
     benchmark.measure_iters=20 benchmark.batch_sizes=[1,2] benchmark.input_sizes=[[256,256]]
   ```
   This catches import errors and checkpoint-load failures before you burn an hour on a full sweep.

## Review checklist

- [ ] Checkpoint loader reused verbatim from the training script.
- [ ] Parity check runs before benchmarks and aborts on failure.
- [ ] Warmup iterations discarded; `measure_iters ≥ 500` for real p99.
- [ ] Timing uses CUDA events, not wall-clock.
- [ ] `nvidia-smi -lgc` wrapped in try/finally.
- [ ] Wandb `job_type="inference-benchmark"` so runs are filterable from training.
- [ ] Peak and steady-state VRAM both logged.
- [ ] Input sizes span the realistic distribution, not one "typical" size.
