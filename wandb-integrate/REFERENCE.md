# Wandb Logging Reference

Detailed code patterns for each of the nine logging categories in [SKILL.md](SKILL.md).

All logs use `step=global_step` (a monotonically-increasing integer counting optimizer steps). `wandb_run` is the return value of `wandb.init()`; guard every log with `if wandb_run:`.

---

## 1. Run metadata (once at start)

Immediately after `wandb.init(...)`:

```python
import socket, subprocess, torch

def _git_info():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
        return {"commit": commit, "branch": branch, "dirty": dirty}
    except Exception:
        return {"commit": None, "branch": None, "dirty": None}

if wandb_run:
    wandb_run.config.update({
        "git": _git_info(),
        "host": socket.gethostname(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
    }, allow_val_change=True)

    # Human-readable name derived from config if not already set
    if not wandb_run.name or wandb_run.name.startswith("run-"):
        wandb_run.name = f"{cfg.model_name}-{cfg.train_config.batch_size}bs-{cfg.train_config.learning_rate:.0e}"
```

Full resolved Hydra config is already passed via `config=OmegaConf.to_container(cfg, resolve=True)` in `wandb.init` — no need to re-log.

---

## 2. Loss & learning signal

### Per step (training)

```python
if wandb_run and global_step % tc.display == 0:
    log = {
        "train/loss": loss.item(),
        "train/epoch": epoch,
    }
    for name, value in loss_components.items():         # multi-task / per-head losses
        log[f"train/loss_{name}"] = float(value)
    wandb_run.log(log, step=global_step)
```

Also accumulate losses into a per-epoch running mean and log it at epoch end under `train/loss_epoch`.

### End of epoch (validation)

```python
val_loss, val_components, val_by_subset, task_metric = run_validation(...)

if wandb_run:
    log = {"val/loss": val_loss, "val/epoch": epoch}
    for name, value in val_components.items():
        log[f"val/loss_{name}"] = float(value)
    for subset, metrics in val_by_subset.items():        # e.g. val/subset_day/iou
        for k, v in metrics.items():
            log[f"val/subset_{subset}/{k}"] = float(v)
    log["val/task_metric"] = float(task_metric)          # accuracy, F1, mAP, perplexity, IoU, etc.
    wandb_run.log(log, step=global_step)
```

**Subset breakdown is mandatory.** Aggregates hide regressions — log `val/subset_<name>/<metric>` for every meaningful data partition (day/night, per-class, per-source, etc.).

---

## 3. Optimization dynamics

Collect inside the training loop, just before and after `optimizer.step()`.

```python
def _total_norm(parameters):
    norms = [p.grad.detach().norm(2) for p in parameters if p.grad is not None]
    return torch.norm(torch.stack(norms), 2).item() if norms else 0.0

def _param_norm(parameters):
    norms = [p.detach().norm(2) for p in parameters]
    return torch.norm(torch.stack(norms), 2).item() if norms else 0.0

# ---- inside loop, after backward(), before optimizer.step() ----
grad_norm_pre = _total_norm(model.parameters())
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=tc.grad_clip)
grad_norm_post = _total_norm(model.parameters())

param_norm_before = _param_norm(model.parameters())
optimizer.step()
param_norm_after = _param_norm(model.parameters())
update_norm = abs(param_norm_after - param_norm_before)

nan_loss = int(torch.isnan(loss).item() or torch.isinf(loss).item())
nan_grads = int(any(
    torch.isnan(p.grad).any() or torch.isinf(p.grad).any()
    for p in model.parameters() if p.grad is not None
))

if wandb_run and global_step % tc.display == 0:
    log = {
        "opt/lr": optimizer.param_groups[0]["lr"],
        "opt/grad_norm_preclip": grad_norm_pre,
        "opt/grad_norm_postclip": grad_norm_post,
        "opt/clip_engaged": int(grad_norm_pre > tc.grad_clip),
        "opt/param_norm": param_norm_after,
        "opt/update_param_ratio": update_norm / max(param_norm_after, 1e-12),  # healthy ≈ 1e-3
        "opt/nan_loss": nan_loss,
        "opt/nan_grads": nan_grads,
    }
    if hasattr(scaler, "get_scale"):            # fp16 / AMP
        log["opt/loss_scale"] = scaler.get_scale()
    wandb_run.log(log, step=global_step)
```

**Update-to-param ratio** is the single most useful signal for LR tuning — values far from 1e-3 (e.g. 1e-5 or 1e-1) mean the LR is miscalibrated for this parameter scale.

---

## 4. Throughput

Time the training loop, split into data-load time and compute time.

```python
import time

data_t0 = time.perf_counter()
for batch in loader:
    data_time = time.perf_counter() - data_t0
    compute_t0 = time.perf_counter()

    # ... forward, backward, step ...

    compute_time = time.perf_counter() - compute_t0
    samples_per_sec = batch_size / (data_time + compute_time)

    if wandb_run and global_step % tc.display == 0:
        wandb_run.log({
            "perf/samples_per_sec": samples_per_sec,
            "perf/data_time_s": data_time,
            "perf/compute_time_s": compute_time,
            "perf/data_compute_ratio": data_time / max(compute_time, 1e-9),  # >0.1 → input-bound
        }, step=global_step)

    data_t0 = time.perf_counter()
```

At end of epoch:

```python
if wandb_run:
    wandb_run.log({"perf/epoch_wall_s": epoch_wall_time, "perf/epoch": epoch}, step=global_step)
```

---

## 5. Hardware

**Do not log by hand.** `wandb.init()` automatically captures GPU utilization, memory (peak + current), power, and temperature under the `system/*` namespace on the run page's System tab. If extra precision is needed:

```python
wandb.init(..., settings=wandb.Settings(_stats_sample_rate_seconds=5))
```

---

## 6. Progress

```python
total_steps = tc.max_epochs * len(train_loader)
frac = global_step / max(total_steps, 1)
elapsed = time.time() - run_start
eta_s = elapsed / max(frac, 1e-9) - elapsed if frac > 0 else 0.0

if wandb_run and global_step % tc.display == 0:
    wandb_run.log({
        "progress/epoch": epoch,
        "progress/fraction": frac,
        "progress/eta_s": eta_s,
    }, step=global_step)
```

---

## 7. Stability tripwires

Track per-epoch aggregates in Python state, log at epoch end.

```python
class StabilityTracker:
    def __init__(self):
        self.nan_this_epoch = False
        self.max_grad = 0.0
        self.max_activation = 0.0
        self.non_improving_epochs = 0
        self.best_val = float("inf")

    def update_step(self, nan_loss, nan_grads, grad_norm_pre, activation_probe):
        if nan_loss or nan_grads:
            self.nan_this_epoch = True
        self.max_grad = max(self.max_grad, grad_norm_pre)
        if activation_probe is not None:
            self.max_activation = max(self.max_activation, float(activation_probe.abs().max()))

    def end_epoch(self, val_loss):
        improved = val_loss < self.best_val - 1e-6
        self.non_improving_epochs = 0 if improved else self.non_improving_epochs + 1
        if improved:
            self.best_val = val_loss

tracker = StabilityTracker()   # construct once before training
```

Register a forward hook on **one fixed probe layer** (choose a deep activation — e.g. the last transformer block's output, or the final backbone feature map) to capture `activation_probe`. Keep the same layer across runs so numbers are comparable.

At epoch end:

```python
if wandb_run:
    wandb_run.log({
        "stability/nan_this_epoch": int(tracker.nan_this_epoch),
        "stability/max_grad_epoch": tracker.max_grad,
        "stability/max_activation_epoch": tracker.max_activation,
        "stability/non_improving_epochs": tracker.non_improving_epochs,
    }, step=global_step)

# reset per-epoch fields
tracker.nan_this_epoch = False
tracker.max_grad = 0.0
tracker.max_activation = 0.0
```

---

## 8. Inference smoke test

A **cheap regression check**, not a benchmark. Catches "inference broke" or "inference got 3× slower" within one epoch.

### Rules

- **Fixed inputs**: 10–50 samples committed to repo at `assets/<MODEL>_smoke/` (identical across runs).
- **Fixed settings**: batch size 1, single input length, fixed seed, deterministic decoding (no sampling).
- **Runs on training hardware**: not a clean benchmark environment — numbers are noisy.
- **Cadence**: once per epoch, after validation.
- **Warmup**: discard the first 2–3 calls before measuring.

### Pattern

```python
from pathlib import Path

def load_smoke_inputs(model_name: str):
    paths = sorted(Path(f"assets/{model_name}_smoke").glob("*.png"))
    return [load_and_preprocess(p) for p in paths]

def run_smoke_test(model, inputs, reference_output, device):
    model.eval()
    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(inputs[0].to(device).unsqueeze(0))
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Measure
    latencies = []
    outputs = []
    with torch.no_grad():
        for x in inputs:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            y = model(x.to(device).unsqueeze(0))
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)
            outputs.append(y.detach().cpu())

    peak_mem = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0

    # Parity check against reference tensor committed to the repo
    diff = (outputs[0] - reference_output).abs().max().item()
    parity_ok = diff < 1e-3

    model.train()
    return {
        "latencies": latencies,
        "outputs": outputs,
        "peak_mem_bytes": peak_mem,
        "parity_ok": parity_ok,
        "parity_max_diff": diff,
    }
```

### Logging

```python
if wandb_run:
    smoke = run_smoke_test(model, smoke_inputs, reference_output, device)

    # Scalars
    wandb_run.log({
        "smoke/latency_mean_s": float(np.mean(smoke["latencies"])),
        "smoke/peak_mem_mb": smoke["peak_mem_bytes"] / 1e6,
        "smoke/parity_ok": int(smoke["parity_ok"]),
        "smoke/parity_max_diff": smoke["parity_max_diff"],
    }, step=global_step)

    # Outputs as a Table — same rows across epochs so you can scrub history
    table = wandb.Table(columns=["input_idx", "output_preview", "latency_s"])
    for i, (out, lat) in enumerate(zip(smoke["outputs"], smoke["latencies"])):
        table.add_data(i, str(out.flatten()[:8].tolist()), lat)
    wandb_run.log({"smoke/outputs": table}, step=global_step)
```

**Scope warning**: `smoke/latency_mean_s` is noisy. p50/p99 and clean-environment numbers belong in a separate post-training benchmark — do not confuse this with that.

---

## Summary block (once at run end)

```python
if wandb_run:
    wandb_run.summary["best_val_loss"] = tracker.best_val
    wandb_run.summary["final_epoch"] = epoch
    wandb_run.summary["total_steps"] = global_step
    wandb_run.finish()
```
