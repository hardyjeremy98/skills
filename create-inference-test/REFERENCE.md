# Inference Test — Reference

Full templates and the reasoning behind each section. SKILL.md has the step-by-step; this file has the code.

## Config template

`src/cable_segmentation/configs/<MODEL>_inference_test.yaml`:

```yaml
# @package _global_
defaults:
  - _self_

checkpoint: ???            # REQUIRED. Pass at CLI: checkpoint=path/to/ckpt
device: cuda
dtype: float32             # float16 / bfloat16 to test cast parity separately

parity:
  num_samples: 16
  atol: 1.0e-4
  rtol: 1.0e-3
  seed: 0

benchmark:
  batch_sizes: [1, 2, 4, 8, 16, 32]
  input_sizes: [[256, 256], [512, 512], [800, 800]]   # (H, W)
  concurrency_levels: [1, 2, 4, 8]
  warmup_iters: 20
  measure_iters: 500       # ≥500 for p99; ≥2000 for p99.9
  lock_gpu_clocks: true    # best effort; non-fatal if it fails

wandb:
  enabled: true
  project: cable-clrnet    # replace per model — see SKILL.md table
  entity: ${oc.env:WANDB_ENTITY,null}
  tags: ["inference-test"]
  job_type: "inference-benchmark"
  name: null               # wandb auto-names if null
  notes: null
```

## Script template

`src/cable_segmentation/<MODEL>_inference_test.py`. Fill in the model-specific `load_model` and `reference_forward` — everything else is mechanical.

```python
#!/usr/bin/env python3
"""<MODEL> post-training inference benchmark.

Runs a correctness parity check against the training-time forward pass, then
measures latency / throughput / VRAM across a batch-size × input-size matrix.
Separate job from training: separate wandb run, separate hardware allocation.
"""

from __future__ import annotations

import logging
import os
import statistics
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def _load_dotenv_for_wandb() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    for path in [*(p / ".env" for p in Path(__file__).resolve().parents), Path.cwd() / ".env"]:
        if path.is_file():
            load_dotenv(path, override=False)
            return


_load_dotenv_for_wandb()


# ---- model-specific: fill these in ----------------------------------------

def load_model(checkpoint: str, device: str, dtype: torch.dtype) -> torch.nn.Module:
    """Load the trained model. Reuse the loader from the training script — do
    not re-implement state_dict handling here."""
    raise NotImplementedError


def reference_forward(model: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    """The training-time forward pass, used as the parity baseline.

    Typically: model.eval(); with torch.no_grad(): return model(inputs)
    If your training eval path does anything different (resize, normalize,
    TTA), reproduce it here."""
    raise NotImplementedError


def build_synthetic_inputs(batch: int, h: int, w: int, device: str, dtype: torch.dtype, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(batch, 3, h, w, generator=g).to(device=device, dtype=dtype)


# ---- generic harness ------------------------------------------------------

@contextmanager
def locked_gpu_clocks(enable: bool):
    locked = False
    if enable:
        try:
            subprocess.run(["nvidia-smi", "-lgc", "1410"], check=True, capture_output=True)  # adjust clock per GPU
            locked = True
            logger.info("GPU clocks locked.")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning("Could not lock GPU clocks: %s", e)
    try:
        yield locked
    finally:
        if locked:
            try:
                subprocess.run(["nvidia-smi", "-rgc"], check=True, capture_output=True)
                logger.info("GPU clocks reset.")
            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to reset GPU clocks: %s", e)


def cuda_time(fn, *args, **kwargs) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn(*args, **kwargs)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)  # ms


def percentile(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def parity_check(model, cfg: DictConfig, device: str, dtype: torch.dtype) -> dict[str, Any]:
    torch.manual_seed(cfg.parity.seed)
    inputs = build_synthetic_inputs(
        batch=cfg.parity.num_samples, h=cfg.benchmark.input_sizes[0][0],
        w=cfg.benchmark.input_sizes[0][1], device=device, dtype=dtype, seed=cfg.parity.seed,
    )
    model.eval()
    with torch.no_grad():
        out_test = model(inputs)
        out_ref = reference_forward(model, inputs)

    if isinstance(out_test, dict):  # handle detection-style outputs
        out_test = next(iter(out_test.values()))
        out_ref = next(iter(out_ref.values())) if isinstance(out_ref, dict) else out_ref

    diff = (out_test.float() - out_ref.float()).abs()
    rel = diff / (out_ref.float().abs() + 1e-8)
    passed = torch.allclose(out_test, out_ref, atol=cfg.parity.atol, rtol=cfg.parity.rtol)
    return {
        "passed": bool(passed),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "max_rel_diff": float(rel.max()),
        "mismatch_fraction": float((diff > cfg.parity.atol).float().mean()),
    }


def benchmark_cell(model, batch: int, h: int, w: int, cfg, device, dtype) -> dict[str, float]:
    inputs = build_synthetic_inputs(batch, h, w, device, dtype, seed=0)
    model.eval()

    for _ in range(cfg.benchmark.warmup_iters):
        with torch.no_grad():
            model(inputs)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    latencies_ms: list[float] = []
    steady_mem: list[int] = []
    for _ in range(cfg.benchmark.measure_iters):
        with torch.no_grad():
            t = cuda_time(model, inputs)
        latencies_ms.append(t)
        steady_mem.append(torch.cuda.memory_allocated())

    return {
        "batch": batch, "h": h, "w": w,
        "p50_ms": percentile(latencies_ms, 0.50),
        "p90_ms": percentile(latencies_ms, 0.90),
        "p99_ms": percentile(latencies_ms, 0.99),
        "mean_ms": statistics.mean(latencies_ms),
        "stddev_ms": statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0.0,
        "throughput_samples_per_s": batch * 1000.0 / statistics.mean(latencies_ms),
        "peak_vram_mb": torch.cuda.max_memory_allocated() / 1e6,
        "steady_vram_mb": statistics.median(steady_mem) / 1e6,
    }


def throughput_under_load(model, batch: int, h: int, w: int, concurrency: int, cfg, device, dtype) -> dict[str, float]:
    inputs = build_synthetic_inputs(batch, h, w, device, dtype, seed=1)
    model.eval()

    def one_request():
        with torch.no_grad():
            start = time.perf_counter()
            model(inputs)
            torch.cuda.synchronize()
            return (time.perf_counter() - start) * 1000

    # warmup
    for _ in range(cfg.benchmark.warmup_iters):
        one_request()

    latencies: list[float] = []
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(one_request) for _ in range(cfg.benchmark.measure_iters)]
        for f in as_completed(futures):
            latencies.append(f.result())
    elapsed = time.perf_counter() - t0
    return {
        "concurrency": concurrency,
        "requests_per_s": len(latencies) / elapsed,
        "p50_ms": percentile(latencies, 0.50),
        "p99_ms": percentile(latencies, 0.99),
    }


@hydra.main(config_path="configs", config_name="<MODEL>_inference_test", version_base="1.3")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    import wandb

    device = cfg.device
    dtype = getattr(torch, cfg.dtype)

    with locked_gpu_clocks(cfg.benchmark.lock_gpu_clocks) as clocks_locked:
        run = wandb.init(
            project=cfg.wandb.project,
            entity=OmegaConf.select(cfg, "wandb.entity"),
            tags=list(cfg.wandb.get("tags", [])),
            job_type=cfg.wandb.job_type,
            name=OmegaConf.select(cfg, "wandb.name"),
            notes=OmegaConf.select(cfg, "wandb.notes"),
            config=OmegaConf.to_container(cfg, resolve=True),
        )

        run.summary["env/torch"] = torch.__version__
        run.summary["env/cuda"] = torch.version.cuda
        run.summary["env/gpu"] = torch.cuda.get_device_name(0)
        run.summary["env/clocks_locked"] = clocks_locked

        model = load_model(cfg.checkpoint, device, dtype)

        # 1) Correctness parity — abort on failure.
        parity = parity_check(model, cfg, device, dtype)
        run.summary.update({f"parity/{k}": v for k, v in parity.items()})
        if not parity["passed"]:
            logger.error("Parity check FAILED: %s. Aborting benchmark.", parity)
            run.alert(title="Inference parity failed", text=str(parity))
            return
        logger.info("Parity check passed (max_abs_diff=%.3e)", parity["max_abs_diff"])

        # 2) Latency matrix.
        cells: list[dict[str, Any]] = []
        for b in cfg.benchmark.batch_sizes:
            for (h, w) in cfg.benchmark.input_sizes:
                try:
                    cell = benchmark_cell(model, b, h, w, cfg, device, dtype)
                except torch.cuda.OutOfMemoryError:
                    logger.warning("OOM at batch=%d input=%dx%d — skipping", b, h, w)
                    torch.cuda.empty_cache()
                    continue
                cells.append(cell)
                run.log({f"latency/b{b}_{h}x{w}/{k}": v for k, v in cell.items() if isinstance(v, (int, float))})

        table = wandb.Table(columns=list(cells[0].keys()), data=[list(c.values()) for c in cells])
        run.log({"latency/matrix": table})

        # 3) Throughput under load — uses the median batch/input cell.
        mid_b = cfg.benchmark.batch_sizes[len(cfg.benchmark.batch_sizes) // 2]
        mid_h, mid_w = cfg.benchmark.input_sizes[len(cfg.benchmark.input_sizes) // 2]
        load_rows = [
            throughput_under_load(model, mid_b, mid_h, mid_w, c, cfg, device, dtype)
            for c in cfg.benchmark.concurrency_levels
        ]
        for row in load_rows:
            run.log({f"load/c{row['concurrency']}/{k}": v for k, v in row.items() if isinstance(v, (int, float))})
        run.log({
            "load/curve": wandb.plot.line(
                wandb.Table(columns=["requests_per_s", "p99_ms"],
                            data=[[r["requests_per_s"], r["p99_ms"]] for r in load_rows]),
                "requests_per_s", "p99_ms", title="Latency vs throughput",
            )
        })

        wandb.finish()


if __name__ == "__main__":
    main()
```

## Why these specific choices

- **CUDA events, not `time.time()`**: `time.time()` captures CPU launch time, not GPU execution. CUDA events sync on the stream and measure the actual kernel time. Wall-clock under-reports for tiny kernels (launch-bound) and over-reports when the CPU is doing anything else.
- **Warmup discarded**: first calls pay for lazy CUDA init, kernel compilation, and cold caches. Including them skews p50 downward and p99 upward in misleading ways.
- **≥500 measure iters**: p99 on 30 samples is basically the max — it has no stable estimator until you have a few hundred samples. p99.9 needs thousands.
- **Parity before speed**: a quantized/compiled/TRT-exported model that gives subtly wrong outputs looks great on a benchmark. The benchmark number is worthless unless correctness is established first.
- **Locked GPU clocks**: unlocked clocks swing ±15% with thermals and background load. The test measures the model, not the thermal state of the box.
- **Separate wandb run (`job_type="inference-benchmark"`)**: keeps inference numbers queryable and filterable. Do not append to the training run — it muddles train/eval dashboards.

## Autoregressive extensions

The models currently in this repo (CLRNet, SegFormer, RF-DETR) are vision models — one forward pass per input, no autoregressive decoding. If the repo ever adds an autoregressive model, extend the script with:

- **Prefill vs decode separation**: prefill processes the prompt (compute-bound, scales with seq length); decode generates one token at a time (memory-bandwidth-bound). Benchmark them separately.
- **TTFT (time-to-first-token)**: prefill latency as experienced by the user. Report per prompt length.
- **ITL (inter-token latency)**: steady-state decode time per token. Usually p50/p99 over the full generation.
- **Input/output length matrix**: prompt_len × output_len, not just batch × shape.
- **Real serving stack**: benchmark vLLM / TGI / TensorRT-LLM, not `model.generate()` in a Python loop. The Python loop is not what ships.
- **KV-cache parity**: caches are a common source of silent correctness bugs — parity-check with and without cache reuse.
- **$/million tokens**: divide cost-per-GPU-hour by sustained throughput at the latency SLO you care about.

Add these as a new section to the config (`autoregressive:`) and new benchmark functions (`benchmark_prefill`, `benchmark_decode`) rather than overloading the vision path.
