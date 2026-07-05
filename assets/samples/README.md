# Sample Rollouts

Sample simulation recordings showing the VLM correction loop in action.

## Task 1: Put the black bowl on top of the cabinet (success)

Three runs of the same task across different experiment iterations, showing the system converging to success in fewer attempts over time.

| Folder | Episode | Solved at attempt | Notes |
|--------|---------|-------------------|-------|
| `20260404_145719/` | ep1585 | a8 | Early experiment |
| `20260407_040404/` | ep1576 | a6 | Improved — 2 fewer attempts |
| `20260415_002747/` | ep1576 | a5 | Further improvement |

Each folder contains `a1` (vanilla VLA, no correction) through the final attempt. Compare `a1` vs the success attempt to see what the VLM correction changed.

## Task 2: Stack the black bowl on the black bowl (failure)

| Folder | Episode | Result | Notes |
|--------|---------|--------|-------|
| `20260416_051438/` | ep840 | all 10 attempts fail | Hard task — stacking requires precise contact geometry |

Included as an honest example of a task the system has not yet solved. The 10 attempts show diverse correction strategies being explored without convergence.
