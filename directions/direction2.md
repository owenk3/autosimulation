# Direction 2: Richer Correction Mechanism

## Objective
Move beyond single time-window corrections to trajectory-based corrections with gripper timing control

## Implementation Requirements
- **Base**: Start from Experiment H successful commit (33.3% baseline)
- **New capabilities**:
  - Multi-timestamp trajectory: corrections at 4-5 key timestamps vs single window
  - Gripper timing: "keep gripper open until t=6s" type controls
  - Still pure action-vector correction (no instruction rewriting, no VLA finetuning)

## Technical Changes
- **Files**: train.py (major restructure)
- **Core changes**:
  - Replace `correction_params` single window with trajectory array
  - Add gripper state control to action vector corrections
  - VLM outputs sequence: [(t1, correction1), (t2, correction2), ...]
- **Complexity**: Hard (60+ lines, new framework)

## Testing Protocol
- Episodes: 15 (consistent with H baseline)
- Success metrics: vs 33.3% Experiment H baseline
- Log file: `experiment_results/direction2.log`
- **Focus**: Multi-step tasks and sequential failures

## Research Question
Can trajectory-based corrections with gripper timing solve multi-step tasks that single-window corrections cannot?

## Status
- [ ] Branch created: `direction2`
- [ ] Implementation started
- [ ] Testing completed
- [ ] Results documented