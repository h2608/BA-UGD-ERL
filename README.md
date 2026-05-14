# BA-UGD-ERL

Minimal research prototype for Budget-Aware Uncertainty-Guided Evolutionary Reinforcement Learning. The current default training path is BA-UGD-ERL with the `ba_scheduler_easy_exploit` scheduler and trajectory filtering, with TD3-only retained as a baseline.

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

BA-UGD-ERL EA rollout smoke test:

```bash
python scripts/smoke_test.py --stage ba_rollout --config configs/hopper.yaml --total_steps 6000
```

BA-UGD-ERL mixed replay smoke test:

```bash
python scripts/smoke_test.py --stage ba_mixed --config configs/hopper.yaml --total_steps 6000
```

BA-UGD-ERL trajectory filter smoke test:

```bash
python scripts/smoke_test.py --stage ba_filter --config configs/hopper.yaml --total_steps 6000
```

BA-UGD-ERL scheduler smoke test:

```bash
python scripts/smoke_test.py --stage ba_scheduler --config configs/hopper.yaml --total_steps 6000
```

BA-UGD-ERL head evolution smoke test:

```bash
python scripts/smoke_test.py --stage ba_evolution --config configs/hopper.yaml --total_steps 6000
```

The TD3 smoke test automatically uses `eval_interval=1000` and one eval episode while keeping `warmup_steps >= 5000`.

## Main Comparison Experiments

The main comparison suite is currently fixed to Hopper-v4 and HalfCheetah-v4, 100k training steps per run, and seeds 0, 1, 2:

```bash
python scripts/run_hopper_experiments.py --config configs/hopper.yaml --total_steps 100000 --seeds 0 1 2
python scripts/run_hopper_experiments.py --config configs/halfcheetah.yaml --total_steps 100000 --seeds 0 1 2
```

Default variants:

- `td3_only`
- `ba_static_switch`
- `ba_scheduler_easy_exploit`
- `ba_scheduler_easy_exploit_no_filter`

Use `--dry_run` to print the planned runs without executing them. Results are appended as JSONL under `outputs/results/`.

`ba_scheduler_easy_exploit` with trajectory filtering is the current main BA-UGD-ERL method. `td3_only`, `ba_static_switch`, and `ba_scheduler_easy_exploit_no_filter` are the fixed main comparison groups.

`ba_static_switch` is implemented through the same discrete scheduler update interval used by the dynamic scheduler. With the current `update_interval=5000`, the realized mode fraction is approximate; for example, a nominal 25% Explore / 75% Exploit switch appears as about 30% / 70% in 50k-100k step runs. Do not report it as a strict continuous 25% / 75% split.

## Training

Run the current main BA-UGD-ERL method:

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

`outputs/models/latest.pt` is a convenience checkpoint for the most recent completed run. Each run also keeps its own checkpoint under `outputs/models/<run_name>/`.

View TensorBoard logs:

```bash
tensorboard --logdir outputs/logs
```

## Current Defaults

- Algorithm: `ba_ugd_erl`
- Main method: `ba_scheduler_easy_exploit` with trajectory filtering
- Warmup steps: `5000`
- Batch size: `128`
- Hidden size: `256`
- Replay capacity: `500000`
- EA heads: `4`
- Trajectory filter margin: `20.0`
- Eval interval: `5000` in normal training, `1000` in smoke tests
- Scheduler update interval: `5000`

## Project Layout

```text
.
|-- configs/
|-- core/
|-- scripts/
|-- outputs/
`-- tests/
```

`core/training.py` is the single training loop used by both `scripts/train.py` and `scripts/smoke_test.py`.
