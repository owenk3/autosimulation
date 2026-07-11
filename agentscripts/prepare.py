"""
prepare.py — episode runner entrypoint.

Wires together: sim, model, vlm, memory modules.
This file is READ-ONLY in the autoresearch loop — the agent modifies train.py.

Usage:
    python agentscripts/prepare.py --task_suite libero_90 --episode 96
    python agentscripts/prepare.py --task_suite libero_90 --eval unified --max_attempts 5
"""

import argparse
import json
import os
import random
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
import yaml

# Repo root on path so all modules resolve
sys.path.insert(0, str(Path(__file__).parent.parent))

# Patch torch.load for LIBERO compatibility (PyTorch 2.6+)
_torch_load_orig = torch.load
torch.load = lambda f, *a, **kw: _torch_load_orig(f, *a, **{**kw, "weights_only": kw.get("weights_only", False)})

from model.vla import get_action, health
from sim.libero import LiberoEnv, load_task, suite_n_tasks

DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")

ACTION_HZ = 30.0
N_ACTION_STEPS = 10
RESIZE_SIZE = 256

_DIM_MAP = {"x": 0, "y": 1, "z": 2, "roll": 3, "pitch": 4, "yaw": 5, "gripper": 6}
_MAGNITUDE_TERM_RATIO = {"little": 0.5, "little_more": 0.75, "slightly": 1.0,
                         "slightly_more": 1.5, "more": 2.0, "much": 4.0}
ACTION_STD = np.array([0.336, 0.378, 0.445, 0.039, 0.063, 0.078, 0.999])


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def resize_image(img):
    from PIL import Image
    return np.array(Image.fromarray(img).resize((RESIZE_SIZE, RESIZE_SIZE)))


def save_rollout_video(frames, episode, success, task_description,
                       model_family="pi05", rollout_dir=None):
    rollout_dir = rollout_dir or f"./rollouts/{DATE}"
    os.makedirs(rollout_dir, exist_ok=True)
    slug = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    path = f"{rollout_dir}/{DATE_TIME}--{model_family}--episode={episode}--success={success}--task={slug}.mp4"
    writer = imageio.get_writer(path, fps=30)
    for img in frames:
        writer.append_data(img)
    writer.close()
    print(f"Saved rollout → {path}")
    return path


def _eval_envelope(t, t_start, t_end, envelope_type="flat"):
    if t_start is None:
        return 1.0
    if t < t_start or t > t_end:
        return 0.0
    if envelope_type == "ramp":
        return (t - t_start) / max(t_end - t_start, 1e-6)
    if envelope_type == "triangle":
        duration = max(t_end - t_start, 1e-6)
        t_mid = (t_start + t_end) / 2.0
        return 2.0 * (t - t_start) / duration if t <= t_mid else 2.0 * (t_end - t) / duration
    ramp = 0.2
    if t <= t_start + ramp:
        return (t - t_start) / ramp
    if t >= t_end - ramp:
        return (t_end - t) / ramp
    return 1.0


def _compile_envelopes(correction_params):
    if not correction_params:
        return []
    compiled = []
    axes = correction_params.get("axes") or [correction_params]
    for ax in axes:
        dim = ax.get("dimension", "unknown")
        if dim not in _DIM_MAP:
            continue
        direction = ax.get("direction", "+1")
        sign = 1.0 if direction == "+1" else -1.0
        mag_val = abs(float(ax.get("magnitude_value") or 0.0))
        term = ax.get("magnitude_term", "unknown")
        magnitude = sign * mag_val if mag_val != 0.0 else (
            sign * _MAGNITUDE_TERM_RATIO[term] * float(ACTION_STD[_DIM_MAP[dim]])
            if term in _MAGNITUDE_TERM_RATIO else sign * 1.0
        )
        mode = ax.get("mode", "add")
        compiled.append({
            "dim_idx": _DIM_MAP[dim],
            "t_start": float(ax["t_start"]) if ax.get("t_start") is not None else None,
            "t_end": float(ax["t_end"]) if ax.get("t_end") is not None else float("inf"),
            "envelope_type": ax.get("envelope", "flat"),
            "magnitude": magnitude,
            "suppress": mode in ("suppress", "override"),
            "add": mode in ("add", "override"),
        })
    return compiled


def run_episode(host, port, env: LiberoEnv, task_description, initial_state,
                seed, num_steps_wait=10, correction_params=None):
    """Run one episode. Returns (success, frames, latency)."""
    t0 = time.time()
    env.reset()
    obs = env.set_init_state(initial_state)

    action_queue = deque()
    frames = []
    envelopes = _compile_envelopes(correction_params)
    set_seed(seed)

    for _ in range(num_steps_wait):
        obs, _, done, _ = env.step(env.dummy_action())
        if done:
            break

    for step in range(env.max_steps):
        frames.append(env.get_image(obs).copy())

        if len(action_queue) == 0:
            observation = {
                "full_image":  resize_image(env.get_image(obs)),
                "wrist_image": resize_image(env.get_wrist_image(obs)),
                "state":       env.get_state(obs),
            }
            action_queue.extend(get_action(host, port, observation, task_description))

        t_now = (step + 1) / ACTION_HZ
        action = action_queue.popleft()

        for cfg in envelopes:
            phi = _eval_envelope(t_now, cfg["t_start"], cfg["t_end"], cfg["envelope_type"])
            if phi == 0.0:
                continue
            di = cfg["dim_idx"]
            if cfg["suppress"]:
                action[di] *= (1.0 - phi)
            if cfg["add"]:
                action[di] += phi * cfg["magnitude"]

        try:
            obs, _, done, _ = env.step(action.tolist())
        except ValueError as e:
            if "terminated episode" in str(e):
                return True, frames, time.time() - t0
            raise
        if done:
            frames.append(env.get_image(obs).copy())
            return True, frames, time.time() - t0

    return False, frames, time.time() - t0


def run_episode_unified(host, port, env, task_description, initial_state,
                        episode, seed, num_steps_wait,
                        max_attempts, video_dir, corrector):
    from agentscripts.train import apply_correction

    prev_frames = []
    for attempt in range(max_attempts):
        print(f"\n[unified] attempt {attempt+1}/{max_attempts}")
        ep_seed = seed + episode - 1

        if attempt == 0:
            success, frames, latency = run_episode(
                host, port, env, task_description, initial_state, ep_seed, num_steps_wait)
            print(f"  vanilla: {'SUCCESS' if success else 'FAILURE'} | {latency:.1f}s")
            save_rollout_video(frames, episode, success, task_description,
                               model_family="pi05_a1", rollout_dir=video_dir)
            if success:
                return True, frames
            prev_frames = frames
        else:
            result = corrector.run(prev_frames, task_description)
            correction = result["correction"]
            effective_task = apply_correction(correction, task_description)
            forced_params = correction.get("params") if correction.get("type") == "action" else None
            print(f"  failure_type={result['failure_type']} from_memory={result['from_memory']}")

            success, frames, latency = run_episode(
                host, port, env, effective_task, initial_state, ep_seed, num_steps_wait,
                correction_params=forced_params)
            print(f"  attempt {attempt+1}: {'SUCCESS' if success else 'FAILURE'} | {latency:.1f}s")
            save_rollout_video(frames, episode, success, task_description,
                               model_family=f"pi05_unified_a{attempt+1}", rollout_dir=video_dir)

            if success:
                corrector.memory.store(task_description, result["failure_type"],
                                       correction, success=True)
                return True, frames
            prev_frames = frames

    return False, prev_frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_suite", default="libero_90")
    parser.add_argument("--episode", type=int, default=-1)
    parser.add_argument("--eval", default="vanilla", choices=["vanilla", "unified"])
    parser.add_argument("--max_attempts", type=int, default=3)
    parser.add_argument("--num_trials_per_task", type=int, default=50)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    host, port = cfg["server"]["host"], cfg["server"]["port"]
    seed = cfg.get("seed", 7)
    num_steps_wait = cfg.get("num_steps_wait", 10)

    # Warmup
    try:
        dummy_obs = {
            "full_image":  np.zeros((256, 256, 3), dtype=np.uint8),
            "wrist_image": np.zeros((256, 256, 3), dtype=np.uint8),
            "state":       np.zeros(8, dtype=np.float32),
        }
        get_action(host, port, dummy_obs, "warmup")
        print("[prepare] Warmup done.")
    except Exception as e:
        print(f"[prepare] Warmup failed: {e}")

    corrector = None
    if args.eval == "unified":
        from agentscripts.train import UnifiedCorrector
        corrector = UnifiedCorrector(memory_dir="./corrections/memory")
        print(f"[unified] memory stats: {corrector.stats()}")

    run_ts = time.strftime("%Y%m%d_%H%M%S")

    episodes = [args.episode] if args.episode > 0 else list(
        range(1, suite_n_tasks(args.task_suite) * args.num_trials_per_task + 1))

    results, total, successes = [], 0, 0

    for episode in episodes:
        task, initial_state, task_id = load_task(
            args.task_suite, episode, args.num_trials_per_task)
        env = LiberoEnv(task, args.task_suite, resolution=cfg.get("env_img_res", 256))

        print(f"\n{'='*60}")
        print(f"[prepare] episode={episode} task_id={task_id}")
        print(f"[prepare] task: {env.task_description}")

        video_dir = f"./rollouts/{args.task_suite}/ep{episode}/{run_ts}"
        os.makedirs(video_dir, exist_ok=True)

        if args.eval == "vanilla":
            success, frames, latency = run_episode(
                host, port, env, env.task_description, initial_state,
                seed + episode - 1, num_steps_wait)
            print(f"[vanilla] {'SUCCESS' if success else 'FAILURE'} | {latency:.1f}s")
            save_rollout_video(frames, episode, success, env.task_description, rollout_dir=video_dir)
        else:
            success, _ = run_episode_unified(
                host, port, env, env.task_description, initial_state,
                episode, seed, num_steps_wait,
                args.max_attempts, video_dir, corrector)

        record = {"eval": args.eval, "episode": episode,
                  "task": env.task_description, "success": success}
        total += 1
        if success:
            successes += 1
        results.append(record)

        with open(os.path.join(video_dir, "result.json"), "w") as f:
            json.dump(record, f, indent=2)

        print(f"[prepare] Running total: {successes}/{total} ({100*successes/total:.1f}%)")
        env.close()

    summary = {
        "eval": args.eval, "suite": args.task_suite,
        "total": total, "successes": successes,
        "success_rate": round(successes / total, 4) if total else 0,
        "episodes": results,
    }
    summary_dir = f"./rollouts/{args.task_suite}/{run_ts}"
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Final: {successes}/{total} = {100*successes/total:.1f}%")
    print(f"Summary → {summary_path}")


if __name__ == "__main__":
    main()
