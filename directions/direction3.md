# Direction 3: VLM Learning Through Accumulated Visual Experience

## Objective
**FLAGSHIP RESEARCH CONTRIBUTION**: VLM learns to correct VLAs through accumulated visual experience vs per-attempt diagnosis

## Implementation Requirements
- **Base**: Start from Experiment H successful commit (33.3% baseline)
- **Components**:
  - **C'**: Soft verification - VLM grades attempts ("80% there, gripper 1cm off") vs binary pass/fail
  - **H**: Depth visualization (already proven for drawer tasks)
  - **J**: Cross-episode memory - share insights across similar task types

## Technical Changes
- **Files**: prepare.py, train.py
- **C' implementation**: Replace binary success with graduated assessment
- **H implementation**: Add depth frames to visual input
- **J implementation**: Extend learnings.json with task category grouping
- **Complexity**: Easy (60 lines total - all Easy components)

## Testing Protocol
- Episodes: 15 (consistent with H baseline)
- Success metrics: vs 33.3% Experiment H baseline
- Log file: `experiment_results/direction3.log`
- **Key metric**: Cross-episode learning effectiveness

## Research Question
Can VLM accumulate visual experience and improve corrections over time vs treating each attempt independently?

## Research Novelty
VLM learning through visual experience accumulation is unexplored - this could be groundbreaking for VLA correction systems.

## Status
- [ ] Branch created: `direction3`
- [ ] Implementation started
- [ ] Testing completed
- [ ] Results documented