# BA-UGD-ERL

Minimal research prototype for Budget-Aware Uncertainty-Guided Evolutionary Reinforcement Learning. Stage A/B currently implements the project skeleton and a TD3-only baseline for MuJoCo continuous-control tasks. BA-UGD-ERL scheduler, EA heads, dual replay buffers, and trajectory filtering are intentionally not implemented yet.

## Environment

Recommended Python version: 3.10.

Create a virtual environment with conda:

```bash
conda env create -f environment.yml
conda activate ba-ugd-erl
```

Or with venv:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For NVIDIA GPUs, install a PyTorch wheel that matches your CUDA driver from the official PyTorch selector if the default pip wheel is not appropriate for your system.

## MuJoCo Notes

This project uses Gymnasium MuJoCo environments and does not require `mujoco-py`.

Minimal environment check:

```bash
python scripts/smoke_test.py --stage env --config configs/hopper.yaml
```

Common fixes:

- If `Hopper-v4` is not registered, reinstall `gymnasium[mujoco]`.
- If MuJoCo native libraries fail to load on Windows, update the GPU driver and install Microsoft C++ Build Tools.
- If rendering fails on a headless machine, keep training/eval render-free; the provided scripts do not request a render mode.
- If package versions conflict, prefer a clean Python 3.10 environment.

## Smoke Tests

Env/import smoke test:

```bash
python scripts/smoke_test.py --stage env --config configs/hopper.yaml
```

TD3-only smoke test:

```bash
python scripts/smoke_test.py --stage td3 --config configs/hopper.yaml --total_steps 6000
```

The TD3 smoke test automatically uses `eval_interval=1000` and one eval episode while keeping `warmup_steps >= 5000`.

## Training

Run TD3-only training:

```bash
python scripts/train.py --config configs/hopper.yaml
```

Override total steps:

```bash
python scripts/train.py --config configs/hopper.yaml --total_steps 50000
```

Evaluate the latest checkpoint:

```bash
python scripts/eval.py --config configs/hopper.yaml --checkpoint outputs/models/latest.pt
```

View TensorBoard logs:

```bash
tensorboard --logdir outputs/logs
```

## Current Defaults

- Algorithm: `td3_only`
- Warmup steps: `5000`
- Batch size: `128`
- Hidden size: `256`
- Replay capacity: `500000`
- Eval interval: `5000` in normal training, `1000` in smoke tests
- Scheduler update interval: reserved as `5000`, disabled in Stage A/B

## Project Layout

```text
.
├─ configs/
├─ core/
├─ scripts/
├─ outputs/
└─ tests/
```

`core/training.py` is the single training loop used by both `scripts/train.py` and `scripts/smoke_test.py`.
