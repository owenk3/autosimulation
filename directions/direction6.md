# Direction 6: Cross-Episode Memory (J*)

## Objective
Develop cross-episode memory alone - share insights across similar task types within one run

## Implementation Requirements
- **Base**: Start from Experiment H successful commit (33.3% baseline)
- **Component**: Multi-episode memory only (J* alone)
- **Method**: Group tasks by category, share learnings across episodes of same type

## Technical Changes
- **Files**: train.py only
- **Core change**: Extend learnings.json grouping by task category
- **Complexity**: Easy (20 lines)

## Testing Protocol
- Episodes: 15 (consistent with H baseline)
- Success metrics: vs 33.3% Experiment H baseline
- Log file: `experiment_results/direction6.log`

## Research Question
Does the system get smarter as it sees more tasks? Can it apply learnings from similar tasks?

## Status
- [ ] Branch created: `direction6`
- [ ] Implementation started
- [ ] Testing completed
- [ ] Results documented