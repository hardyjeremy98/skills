---
name: wandb-integrate
description: Wire a model training script into Weights & Biases with a comprehensive logging spec (loss, optimization dynamics, throughput, stability tripwires, inference smoke test, run metadata). Use when adding wandb to a new/existing trainer, or when a user asks for "wandb integration", "wandb logging", or "wandb setup" for a specific model.
---

# Wandb Integrate

Wire a model's training script to Weights & Biases with a complete logging spec.

## Step 1 — Confirm the model

The user must name the model being integrated (e.g. `clrnet`, `segformer`, `rfdetr`). If they did not supply one, **ask first**:

> "Which model is this wandb integration for? I'll use the name for the project (default: `<model>`) and filenames."

Use `<MODEL>` as the placeholder below.

## Step 2 — Environment & config

**Env file** (`.env` at repo root):
```
WANDB_API_KEY=...
WANDB_ENTITY=...           # optional
WANDB_PROJECT=<MODEL>      # optional; overrides config default
```

**Hydra block** — add to the model's root YAML:
```yaml
wandb:
  enabled: true
  project: ${oc.env:WANDB_PROJECT,<MODEL>}   # defaults to model name, overridable via env
  entity: ${oc.env:WANDB_ENTITY,null}
  name: null                                  # set at runtime from config hash / timestamp
  tags: []
  group: null
  notes: null
```

**Init pattern** (in training entrypoint, before any log call):
```python
from dotenv import load_dotenv
load_dotenv()                                 # loads WANDB_API_KEY into env

import wandb
from omegaconf import OmegaConf

wandb_run = wandb.init(
    project=cfg.wandb.project,
    entity=OmegaConf.select(cfg, "wandb.entity"),
    name=OmegaConf.select(cfg, "wandb.name"),
    tags=list(cfg.wandb.get("tags", [])),
    group=OmegaConf.select(cfg, "wandb.group"),
    notes=OmegaConf.select(cfg, "wandb.notes"),
    config=OmegaConf.to_container(cfg, resolve=True),
)
```

## Step 3 — What to log

Eight categories. Each has code snippets and key names in [REFERENCE.md](REFERENCE.md).

| # | Category | When | Where to look |
|---|---|---|---|
| 1 | **Run metadata** (config, git, host, device, world size, run name, tags, group, notes) | Once at start | [REFERENCE.md § 1](REFERENCE.md#1-run-metadata-once-at-start) |
| 2 | **Loss & learning signal** (train loss per step + epoch, per-component, val loss, val metrics by subset, task metric) | Per step + end of epoch | [§ 2](REFERENCE.md#2-loss--learning-signal) |
| 3 | **Optimization dynamics** (LR, grad-norm pre/post-clip, param-norm, update-to-param ratio, NaN/Inf counts, loss scale) | Per step | [§ 3](REFERENCE.md#3-optimization-dynamics) |
| 4 | **Throughput** (samples/sec, epoch wall-clock, data-load vs compute ratio) | Per step + end of epoch | [§ 4](REFERENCE.md#4-throughput) |
| 5 | **Hardware** (GPU util/mem/power/temp) | Automatic — do not log by hand | [§ 5](REFERENCE.md#5-hardware) |
| 6 | **Progress** (epoch, fraction complete, ETA) | Per step | [§ 6](REFERENCE.md#6-progress) |
| 7 | **Stability tripwires** (NaN-this-epoch bool, max grad/activation magnitude, consecutive non-improving epochs) | Per step + end of epoch | [§ 7](REFERENCE.md#7-stability-tripwires) |
| 8 | **Inference smoke test** (fixed inputs → latency, peak mem, output table, parity diff) | Once per epoch after val | [§ 8](REFERENCE.md#8-inference-smoke-test) |

## Step 4 — Cadence rules

- Use `step=global_step` on every `wandb_run.log()` — never epoch. Curves must share one x-axis.
- **Per-step** (inside training loop): categories 2 (train loss), 3, 4, 6, 7.
- **End of epoch**: categories 2 (validation + task metric), 4 (epoch wall-clock), 7 (epoch aggregates), 8.
- **Once at start**: category 1.
- Guard every log call with `if wandb_run:` — training must still run when wandb is disabled.

## Step 5 — Smoke test setup

A cheap per-epoch regression check. See [REFERENCE.md § 8](REFERENCE.md#8-inference-smoke-test) for the full code pattern. Key rules:

- Fixed inputs committed to repo at `assets/<MODEL>_smoke/` (10–50 samples).
- Batch size 1, single length, fixed seed, deterministic decoding.
- Discard first 2–3 warmup calls before measuring latency.
- Log mean latency, peak memory, outputs as `wandb.Table` (scrubbable across epochs), and a pass/fail parity check vs a reference output with max diff.

## Review checklist

Before marking integration complete:

- [ ] Run `cable-seg-<MODEL>-train` (or equivalent) and confirm the wandb run page shows all 8 categories.
- [ ] Kill the run after 2 epochs and verify the epoch-end logs fired (val, smoke test, tripwire aggregates).
- [ ] Disable wandb (`wandb.enabled=false`) and confirm training still runs cleanly — no `wandb_run` attribute errors.
- [ ] Open the run page: config tab shows full resolved Hydra config; system tab shows GPU metrics.
- [ ] Verify `WANDB_API_KEY` is loaded from `.env`, not hardcoded or prompted at runtime.
