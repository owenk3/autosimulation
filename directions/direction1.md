# Direction 1: G+H+I Integration

## Objective
Combine three proven experiments: G (Correction Templates), H (Depth Visualization), I (Action-Conditioned)

## Implementation Requirements
- **Base**: Start from Experiment H successful commit (33.3% baseline)
- **Components to integrate**:
  - **G**: Task category classifier + correction templates (Medium - 30 lines)
  - **H**: Depth visualization with cv2.applyColorMap() (Easy - 15 lines)
  - **I**: VLA action logging + analysis (Medium - 30 lines)

## Technical Changes
- **Files**: prepare.py, train.py
- **Integration points**:
  - prepare.py: Save depth + log VLA actions
  - train.py: Task classifier → templates + depth frames + action analysis
- **Complexity**: Medium (75+ lines total)

## Testing Protocol
- Episodes: 15 (consistent with H baseline)
- Success metrics: vs 33.3% Experiment H baseline
- Log file: `experiment_results/direction1.log`

## Research Question
Do the three components (templates + depth + action analysis) provide additive benefits?

## Status
- [ ] Branch created: `direction1`
- [ ] Implementation started
- [ ] Testing completed
- [ ] Results documented