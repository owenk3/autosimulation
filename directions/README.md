# directions/

This folder is the **research archive** for the direction evaluation phase. It contains the frozen baseline code and the six research briefs that were tested against it.

---

## What's in here

```
directions/
  train_baseline.py   — frozen snapshot of train.py at 36.0% success (Direction 3 winner)
  direction1.md       — research brief: G+H+I integration
  direction2.md       — research brief: richer trajectory corrections
  direction3.md       — research brief: VLM visual experience learning (won, became baseline)
  direction4.md       — research brief: soft verification alone
  direction5.md       — research brief: depth visualization alone
  direction6.md       — research brief: cross-episode memory alone
```

### `train_baseline.py`
The **official 36.0% baseline** — a frozen, runnable copy of `train.py` as it was when Direction 3 achieved 36% success (18/50 episodes). Never modify this file. Use it to:
- Reproduce the baseline result to verify your setup
- Compare any new `train.py` changes against a known reference
- Roll back if a new approach regresses below 36%

The current `prepare.py` in the root is identical to what was used at this baseline — no separate `prepare_baseline.py` is needed.

### `direction{N}.md`
Research briefs written before each direction was tested. Each brief describes:
- The hypothesis and objective
- Specific code changes required
- Which components (C', H, J, etc.) to implement
- The research question being answered

These are historical — all six directions have been completed. Direction 3 (C'+H+J integration) won at 36.0% and became the current baseline. The briefs are kept as a record of what was tried and why.

---

## Direction results summary

| Direction | Strategy | Result vs 33.3% baseline |
|-----------|----------|--------------------------|
| Direction 1 | G+H+I integration | below baseline |
| Direction 2 | Richer trajectory corrections | below baseline |
| **Direction 3** | **VLM visual experience (C'+H+J)** | **36.0% ✓ winner** |
| Direction 4 | Soft verification alone | below baseline |
| Direction 5 | Depth visualization alone | below baseline |
| Direction 6 | Cross-episode memory alone | below baseline |

---

## How to reproduce the 36% baseline

Use this to verify your environment is set up correctly before running new experiments.

**Step 1 — SSH tunnel** (must be active before running):
```bash
ssh -o ServerAliveInterval=30 -f -N -L ${SSH_LOCAL_PORT}:localhost:${SSH_LOCAL_PORT} \
    -i ${SSH_KEY} ${SSH_USER}@${SSH_HOST}
```

**Step 2 — Environment**:
```bash
conda activate autosim
export PYTHONPATH=/private/tmp/LIBERO:$PYTHONPATH
export MUJOCO_GL=cgl
```

**Step 3 — Swap in the baseline `train.py`**:
```bash
cp directions/train_baseline.py train.py
```

**Step 4 — Run initial validation** (50 episodes, fixed seed):
```bash
PYTHONUNBUFFERED=1 python prepare.py --task_suite libero_90 --eval unified --mode initial
```

Expected output: `18/50 = 36.0%` success rate.

**Step 5 — Restore your current `train.py`**:
```bash
git checkout train.py
```

---

## How to test a new direction

If you want to test a new research direction against the baseline:

1. **Create a branch** from the current baseline commit:
   ```bash
   git checkout -b direction7
   ```

2. **Modify `train.py` only** — `prepare.py` is read-only and must not be changed.

3. **Run initial validation**:
   ```bash
   PYTHONUNBUFFERED=1 python prepare.py --task_suite libero_90 --eval unified --mode initial \
       2>&1 | tee experiment_results/direction7.log
   ```

4. **Compare against baseline** — look for improvement over 36.0%.

5. **Log result** to `results.tsv`:
   ```
   [commit_hash]   [success_rate]   [keep/discard]   [description]
   ```

6. **Document** what changed and why in `experiments.md`.

Only run full validation (`--mode full`, all 4500 episodes) after initial validation shows a clear improvement.
