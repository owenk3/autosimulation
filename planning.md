-Auto Simulation Resarch

Current Process
	1. Failure Video here - Analyze failure video - Make Instruction 
	2. Test on remote server using made instruction from (1) and see the result. 
	3. Human observe the reuslt, if failed, repeat from process 1.

New Process
    1. Automate all processes here and continously develop instructions.


Need to check
[General]
-all environment will use conda environment autosim and version controlled by uv


[Detail]
1. System Setup
    -pi 0.5 model load at this laptop
    -Simulation Dir: LIBERO: ./simulations/ 

┌─── Your Mac (native ARM64, no Docker needed) ────────┐
│                                                        │
│  LIBERO/robosuite simulation (MuJoCo has ARM64 wheels) │
│  AI Agent loop (autosimulation orchestrator)           │
│  Video analysis (Cluade code)                       │
│                                                        │
│  ──── HTTP call (~60-100ms round trip) ────────────►   │
└────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─── RunPod/Modal Serverless GPU ($0.20-0.35/hr) ──────┐
│                                                        │
│  Pi 0.5 inference server (openpi serve_policy)        │
│  ~40-73ms per inference step                          │
│                                                        │
└────────────────────────────────────────────────────────┘

2. Code Setup
