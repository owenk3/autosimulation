# Direction 5: Depth Visualization (H*)

## Objective
Develop depth visualization alone - already proven to solve drawer insertion tasks

## Implementation Requirements
- **Base**: Start from Experiment H successful commit (33.3% baseline)
- **Component**: Depth visualization only (H* alone)
- **Method**: cv2.applyColorMap(depth) frames sent to VLM

## Technical Changes
- **Files**: prepare.py, train.py
- **Core change**: Add depth-colorized frames alongside RGB
- **Complexity**: Easy (15 lines)

## Testing Protocol
- Episodes: 15 (consistent with H baseline)
- Success metrics: vs 33.3% Experiment H baseline
- Log file: `experiment_results/direction5.log`

## Research Question
How much does depth help across diverse tasks beyond drawer insertion?

## Status
- [ ] Branch created: `direction5`
- [ ] Implementation started
- [ ] Testing completed
- [ ] Results documented