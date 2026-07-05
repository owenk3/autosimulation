"""
prepare.py — LIBERO episode runner for autosimulation.

Runs pi05 episodes via remote model server, saves rollout videos.
This file is READ-ONLY in the autoresearch loop — the agent modifies train.py.

Usage:
    python prepare.py --task_suite libero_90 --episode 96
    python prepare.py --task_suite libero_spatial --episode 1 --eval unified --max_attempts 3
"""

import argparse
import json
import math
import os
import random
import time
from collections import deque
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
import yaml

# Patch torch.load for LIBERO compatibility (PyTorch 2.6+ defaults to weights_only=True)
_torch_load_orig = torch.load
torch.load = lambda f, *a, **kw: _torch_load_orig(f, *a, **{**kw, "weights_only": kw.get("weights_only", False)})

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from model_client import get_action

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")

TASK_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}

N_ACTION_STEPS = 10
RESIZE_SIZE = 256
ACTION_HZ = 30.0

_DIM_MAP = {"x": 0, "y": 1, "z": 2, "roll": 3, "pitch": 4, "yaw": 5, "gripper": 6}
_MAGNITUDE_TERM_RATIO = {"little": 0.5, "little_more": 0.75, "slightly": 1.0,
                         "slightly_more": 1.5, "more": 2.0, "much": 4.0}
ACTION_STD = np.array([0.336, 0.378, 0.445, 0.039, 0.063, 0.078, 0.999])


# ---------------------------------------------------------------------------
# Seed / utils
# ---------------------------------------------------------------------------

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def resize_image(img):
    from PIL import Image
    return np.array(Image.fromarray(img).resize((RESIZE_SIZE, RESIZE_SIZE)))


def process_action(action):
    return action


def decode_mp4_frames(mp4_path):
    cap = cv2.VideoCapture(mp4_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def episode_to_task(episode, num_trials_per_task=50):
    idx = episode - 1
    return idx // num_trials_per_task, idx % num_trials_per_task


# ---------------------------------------------------------------------------
# LIBERO helpers (from libero_utils.py)
# ---------------------------------------------------------------------------

def get_libero_env(task, model_family, resolution=256):
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    return env, task_description


def get_libero_dummy_action(model_family):
    return [0, 0, 0, 0, 0, 0, -1]


def get_libero_image(obs):
    return obs["agentview_image"][::-1, ::-1]


def get_libero_wrist_image(obs):
    return obs["robot0_eye_in_hand_image"][::-1, ::-1]


def quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def save_rollout_video(rollout_images, idx, success, task_description, model_family="pi05", rollout_dir=None):
    if rollout_dir is None:
        rollout_dir = f"./rollouts/{DATE}"
    os.makedirs(rollout_dir, exist_ok=True)
    task_slug = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = f"{rollout_dir}/{DATE_TIME}--{model_family}--episode={idx}--success={success}--task={task_slug}.mp4"
    writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        writer.append_data(img)
    writer.close()
    print(f"Saved rollout → {mp4_path}")
    return mp4_path


# ---------------------------------------------------------------------------
# Action bias (from world.py + action_bias.py)
# ---------------------------------------------------------------------------

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
        if mag_val != 0.0:
            magnitude = sign * mag_val
        elif term in _MAGNITUDE_TERM_RATIO:
            magnitude = sign * _MAGNITUDE_TERM_RATIO[term] * float(ACTION_STD[_DIM_MAP[dim]])
        else:
            magnitude = sign * 1.0
        mode = ax.get("mode", "add")
        compiled.append({
            "dim_idx": _DIM_MAP[dim], "dim": dim,
            "t_start": float(ax["t_start"]) if ax.get("t_start") is not None else None,
            "t_end": float(ax["t_end"]) if ax.get("t_end") is not None else float("inf"),
            "envelope_type": ax.get("envelope", "flat"),
            "magnitude": magnitude,
            "suppress": mode in ("suppress", "override"),
            "add": mode in ("add", "override"),
        })
    return compiled


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(host, port, env, task_description, initial_state,
                task_suite, seed, num_steps_wait=10, correction_params=None):
    """Run one LIBERO episode using remote model server. Returns (success, frames, latency)."""
    t0 = time.time()
    env.reset()
    obs = env.set_init_state(initial_state)

    action_queue = deque()
    replay_images = []
    max_steps = TASK_MAX_STEPS[task_suite]
    success = False
    step_count = 0

    set_seed(seed)
    envelopes = _compile_envelopes(correction_params)

    # Wait steps (with done check, matching original)
    for _ in range(num_steps_wait):
        obs, _, done, _ = env.step(get_libero_dummy_action("pi05"))
        if done:
            break

    for _ in range(max_steps):
        img = get_libero_image(obs)
        wrist_img = get_libero_wrist_image(obs)
        replay_images.append(img.copy())

        if len(action_queue) == 0:
            observation = {
                "full_image": resize_image(img),
                "wrist_image": resize_image(wrist_img),
                "state": np.concatenate((
                    obs["robot0_eef_pos"],
                    quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                )),
            }
            actions = get_action(host, port, observation, task_description)
            action_queue.extend(actions)

        step_count += 1
        t_now = step_count / ACTION_HZ

        action = process_action(action_queue.popleft())

        # Apply action bias
        for env_cfg in envelopes:
            phi = _eval_envelope(t_now, env_cfg["t_start"], env_cfg["t_end"], env_cfg["envelope_type"])
            if phi == 0.0:
                continue
            di = env_cfg["dim_idx"]
            if env_cfg["suppress"]:
                action[di] *= (1.0 - phi)
            if env_cfg["add"]:
                action[di] += phi * env_cfg["magnitude"]

        try:
            obs, _, done, _ = env.step(action.tolist())
        except ValueError as e:
            if "terminated episode" in str(e):
                success = True
                break
            raise
        if done:
            success = True
            replay_images.append(get_libero_image(obs).copy())
            break

    return success, replay_images, time.time() - t0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_single_episode_eval(host, port, env, task_description, initial_state,
                            suite_name, episode, seed, num_steps_wait, eval_mode,
                            max_attempts, video_dir, corrector=None):
    """Run one episode (vanilla or unified). Returns (success, result_dict)."""
    set_seed(seed + episode - 1)

    if eval_mode == "vanilla":
        success, frames, latency = run_episode(
            host, port, env, task_description, initial_state,
            suite_name, seed + episode - 1, num_steps_wait)
        print(f"[vanilla] {'SUCCESS' if success else 'FAILURE'} | {latency:.1f}s")
        save_rollout_video(frames, episode, success, task_description, rollout_dir=video_dir)
        return success, {"eval": eval_mode, "episode": episode, "task": task_description, "success": success}

    # unified
    from train import apply_correction
    prev_frames = []
    success = False

    for attempt in range(max_attempts):
        print(f"\n[unified] attempt {attempt+1}/{max_attempts}")

        if attempt == 0:
            success, frames, latency = run_episode(
                host, port, env, task_description, initial_state,
                suite_name, seed + episode - 1, num_steps_wait)
            print(f"  vanilla: {'SUCCESS' if success else 'FAILURE'} | {latency:.1f}s")
            save_rollout_video(
                frames, episode, success, task_description,
                model_family=f"pi05_a{attempt+1}", rollout_dir=video_dir)
            if success:
                break
            prev_frames = frames
        else:
            result = corrector.run(prev_frames, task_description)
            correction = result["correction"]
            effective_task = apply_correction(correction, task_description)
            forced_params = correction.get("params") if correction.get("type") == "action" else None
            print(f"  failure_type={result['failure_type']} from_memory={result['from_memory']}")
            print(f"  effective_task: {effective_task}")
            if forced_params:
                print(f"  correction_params: {forced_params}")

            success, frames, latency = run_episode(
                host, port, env, effective_task, initial_state,
                suite_name, seed + episode - 1, num_steps_wait,
                correction_params=forced_params)
            print(f"  attempt {attempt+1}: {'SUCCESS' if success else 'FAILURE'} | {latency:.1f}s")
            save_rollout_video(
                frames, episode, success, task_description,
                model_family=f"pi05_unified_a{attempt+1}", rollout_dir=video_dir)

            if success:
                corrector.memory.store(task_description, result["failure_type"], correction, success=True)
                break
            prev_frames = frames

    return success, {"eval": eval_mode, "episode": episode, "task": task_description, "success": success}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_suite", default="libero_90")
    parser.add_argument("--episode", type=int, default=-1, help="1-indexed global episode (-1 = all)")
    parser.add_argument("--eval", default="vanilla", choices=["vanilla", "unified"])
    parser.add_argument("--max_attempts", type=int, default=3)
    parser.add_argument("--num_trials_per_task", type=int, default=50)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    host, port = cfg["server"]["host"], cfg["server"]["port"]
    seed = cfg.get("seed", 7)
    num_steps_wait = cfg.get("num_steps_wait", 10)

    # Warm up model server
    print("[prepare] Warming up model server...")
    try:
        dummy_obs = {
            "full_image": np.zeros((256, 256, 3), dtype=np.uint8),
            "wrist_image": np.zeros((256, 256, 3), dtype=np.uint8),
            "state": np.zeros(8, dtype=np.float32),
        }
        get_action(host, port, dummy_obs, "warmup")
        print("[prepare] Warmup done.")
    except Exception as e:
        print(f"[prepare] Warmup failed (first real call may be slow): {e}")

    # Setup corrector once (shared across episodes)
    corrector = None
    if args.eval == "unified":
        from train import UnifiedCorrector
        corrector = UnifiedCorrector(memory_dir=str(Path("./corrections/memory")))
        print(f"[unified] memory stats: {corrector.stats()}")

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite]()
    run_ts = time.strftime("%Y%m%d_%H%M%S")

    # Build episode list
    if args.episode > 0:
        episodes = [args.episode]
    else:
        # All episodes: n_tasks * num_trials_per_task, 1-indexed
        episodes = list(range(1, task_suite.n_tasks * args.num_trials_per_task + 1))

    print(f"[prepare] suite={args.task_suite} eval={args.eval} episodes={len(episodes)}")

    results = []
    total, successes = 0, 0

    for episode in episodes:
        task_id, episode_idx = episode_to_task(episode, args.num_trials_per_task)
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, "pi05", resolution=cfg.get("env_img_res", 256))
        initial_state = initial_states[episode_idx]

        print(f"\n{'='*60}")
        print(f"[prepare] episode={episode} task_id={task_id} ep_idx={episode_idx}")
        print(f"[prepare] task: {task_description}")

        video_dir = f"./rollouts/{args.task_suite}/ep{episode}/{run_ts}"
        os.makedirs(video_dir, exist_ok=True)

        success, record = run_single_episode_eval(
            host, port, env, task_description, initial_state,
            args.task_suite, episode, seed, num_steps_wait,
            args.eval, args.max_attempts, video_dir, corrector)

        total += 1
        if success:
            successes += 1
        results.append(record)

        # Save per-episode result
        with open(os.path.join(video_dir, "result.json"), "w") as f:
            json.dump(record, f, indent=2)

        print(f"[prepare] Running total: {successes}/{total} ({100*successes/total:.1f}%)")

        env.close()

    # Save batch summary
    summary = {
        "eval": args.eval,
        "suite": args.task_suite,
        "total": total,
        "successes": successes,
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
