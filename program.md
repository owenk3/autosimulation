# Autosimulation 
This is an experiment to have VLM-VLA do its own research in simulation setup.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autosimulation/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autosimulation/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — constants, simulation prep + runtime utilities. Do not modify.
   - `train.py` — the file you modify. VLM Instruction for in-context learning.
   - `config.yaml` — fixed environment config (read-only)
   - `rollouts/` — saved videos for analysis

4. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
5. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: 

# 1. conda enviornment here
conda activate autosim 
export PYTHONPATH=/private/tmp/LIBERO:$PYTHONPATH
export MUJOCO_GL=cgl

# 2. Install dependencies
uv sync

# 3. Manually run a single training experiment (~5 min)
python prepare.py --task_suite libero_90 --eval unified

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: task instruction for VLM.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_bpb` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the higest val_success_rate.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change instructions for VLM. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful val_success_rate gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A few val_success_rate improvement that adds 20 lines of hacky code? Probably not worth it. A few val_success_rate improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.


## Output format

Once the script finishes it prints a summary like this:

```
---
val_sucess_rate:  0.92
training_seconds: 300.1
total_seconds:    325.9
```

Note that the script is configured to always stop after 5 minutes, so depending on the computing platform of this computer the numbers might look different. You can extract the key metric from the log file:

```
grep "^val_success_rate:" run.log
```


## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	val_sucess_rate	  status	description
```

1. git commit hash (short, 7 chars)
2. val_sucess_rate achieved (e.g. 0.92234567) — use 0.000000 for crashes
3. status: `keep`, `discard`, or `crash`
4. short text description of what this experiment tried

Example:

```
commit	val_bpb		status	description
a1b2c3d	0.997900	keep	baseline
b2c3d4e	0.993200	keep	increase .. to 0.04
c3d4e5f	1.005000	discard	switch to ..
d4e5f6g	0.000000	crash	(OOM)
```


## The experiment loop

The experiment runs on a dedicated branch (e.g. `autosimulation/mar5` or `autosimulation/mar5-gpu0`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^val_sucess_Rate" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If val_sucess_rate improved (lower), you "advance" the branch, keeping the git commit
9. If val_sucess_rate is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!


