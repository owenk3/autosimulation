# Direction 4: Soft Verification (C*)

## Objective
Develop graduated self-assessment alone - VLM grades each attempt on a scale vs binary pass/fail

## Implementation Requirements
- **Base**: Start from Experiment H successful commit (33.3% baseline)
- **Component**: Soft verification only (C* alone)
- **Method**: VLM rates "how close" each attempt gets, feeds into next attempt reasoning

## Technical Changes
- **Files**: train.py, prepare.py
- **Core change**: Replace _verify_success binary with graduated assessment
- **Complexity**: Easy (20 lines - modified C experiment)

## Testing Protocol
- Episodes: 15 (consistent with H baseline)
- Success metrics: vs 33.3% Experiment H baseline
- Log file: `experiment_results/direction4.log`

## Research Question
Does graduated self-assessment improve convergence vs binary pass/fail?

## Status
- [ ] Branch created: `direction4`
- [ ] Implementation started
- [ ] Testing completed
- [ ] Results documented