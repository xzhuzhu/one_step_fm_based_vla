"""VLA-Adapter-derived LIBERO evaluation for a TurboVLA checkpoint.

This script intentionally reuses the VLA-Adapter task/episode protocol shape,
but bypasses OpenVLA/Prismatic preprocessing and action unnormalization. The
policy adapter keeps GroundingDINO's 256px DINOv3 inputs, proprio stats, hard
action min/max, and gripper sign rule.

VLA-Adapter is MIT-licensed; see ../LICENSES/VLA-Adapter.txt.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
from dataclasses import dataclass, fields
import json
import logging
import os
from pathlib import Path
from typing import ClassVar

from turbovla.evaluation.protocol import (
    CHUNK_SIZE, IMAGE_SIZE, OPEN_LOOP_STEPS, PRECISION, PROTOCOL, SEED,
    SETTLE_STEPS, SHARDS, SUITES, TRIALS_PER_TASK,
)

import imageio
import numpy as np
import tqdm


TASK_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}

LIBERO_SUITES = SUITES


@dataclass
class GenerateConfig:
    """Internal worker paths and shard identity; protocol values are fixed."""

    ckpt_path: str = ""
    dinov3_path: str = ""
    bert_path: str = ""
    r3m_path: str = ""
    stats_path: str = "experiments/libero/configs/libero_all4_stats.json"
    task_suite_name: str = "libero_10"
    eval_episode_slice_start: int = 0
    save_video: bool = False
    video_out_path: str = "outputs/evaluation/videos"
    result_json_path: str = ""
    log_path: str = ""
    dry_run_model_load: bool = False

    stats_key: ClassVar[str] = "libero_all4_no_noops"
    num_trials_per_task: ClassVar[int] = TRIALS_PER_TASK
    eval_episode_slice_stride: ClassVar[int] = SHARDS
    num_steps_wait: ClassVar[int] = SETTLE_STEPS
    num_open_loop_steps: ClassVar[int] = OPEN_LOOP_STEPS
    env_img_res: ClassVar[int] = IMAGE_SIZE
    seed: ClassVar[int] = SEED
    chunk_size: ClassVar[int] = CHUNK_SIZE
    precision: ClassVar[str] = PRECISION


def _parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).lower()
    if lowered in {"1", "true", "t", "yes", "y"}:
        return True
    if lowered in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def parse_args() -> GenerateConfig:
    parser = argparse.ArgumentParser(
        description="VLA-Adapter-derived aligned LIBERO evaluation for TurboVLA checkpoints."
    )
    for field in fields(GenerateConfig):
        name = field.name
        default = field.default
        arg_name = f"--{name}"
        dashed_arg_name = f"--{name.replace('_', '-')}"
        kwargs = {"default": default, "help": f"default: {default}"}
        if isinstance(default, bool):
            parser.add_argument(arg_name, dashed_arg_name, type=_parse_bool, nargs="?", const=True, **kwargs)
            parser.add_argument(f"--no_{name}", f"--no-{name.replace('_', '-')}", dest=name, action="store_false")
        elif isinstance(default, int):
            parser.add_argument(arg_name, dashed_arg_name, type=int, **kwargs)
        elif isinstance(default, float):
            parser.add_argument(arg_name, dashed_arg_name, type=float, **kwargs)
        else:
            parser.add_argument(arg_name, dashed_arg_name, type=str, **kwargs)
    return GenerateConfig(**vars(parser.parse_args()))


def _import_turbovla_adapter():
    from turbovla.evaluation.suite_policy import (
        TurboVLAPolicy,
        get_libero_dummy_action,
        rotate_libero_image,
        set_seed_everywhere,
    )

    return TurboVLAPolicy, get_libero_dummy_action, rotate_libero_image, set_seed_everywhere


def _setup_logging(cfg: GenerateConfig) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if cfg.log_path:
        Path(cfg.log_path).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(cfg.log_path, mode="w", encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
        force=True,
    )


def _make_libero_env(task, cfg: GenerateConfig):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_description = task.language
    task_bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": str(task_bddl_file),
        "camera_heights": cfg.env_img_res,
        "camera_widths": cfg.env_img_res,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(cfg.seed)
    return env, task_description


def _save_video(path: Path, frames: list[np.ndarray], fps: int = 20) -> None:
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(path, [np.asarray(x) for x in frames], fps=fps)


def _result_path(cfg: GenerateConfig) -> Path:
    if cfg.result_json_path:
        return Path(cfg.result_json_path)
    ckpt_tag = Path(cfg.ckpt_path).stem
    return Path(cfg.video_out_path) / f"{ckpt_tag}_{cfg.task_suite_name}_results.json"


def _run_episode(
    cfg: GenerateConfig,
    env,
    policy,
    task_description: str,
    initial_state: np.ndarray,
    get_libero_dummy_action,
    rotate_libero_image,
) -> tuple[bool, list[np.ndarray], list[int]]:
    env.reset()
    obs = env.set_init_state(initial_state)
    policy.reset_history()
    action_queue: deque[np.ndarray] = deque()
    replay_images: list[np.ndarray] = []
    query_horizons: list[int] = []
    dummy_action = np.asarray(get_libero_dummy_action(), dtype=np.float32)
    for t in range(TASK_MAX_STEPS[cfg.task_suite_name] + SETTLE_STEPS):
        if t < SETTLE_STEPS:
            obs, _, _, _ = env.step(dummy_action.tolist())
            continue
        if not action_queue or cfg.save_video:
            primary = rotate_libero_image(obs["agentview_image"])
            wrist = rotate_libero_image(obs["robot0_eye_in_hand_image"])
            if cfg.save_video:
                replay_images.append(np.concatenate([primary, wrist], axis=1))
        if not action_queue:
            env_actions = policy.predict_env_action_chunk(
                primary, wrist, task_description, obs, execute_steps=OPEN_LOOP_STEPS,
            )
            if len(env_actions) != OPEN_LOOP_STEPS:
                raise RuntimeError("policy must supply ten finite actions per query")
            query_horizons.append(OPEN_LOOP_STEPS)
            action_queue.extend(np.asarray(row, dtype=np.float32) for row in env_actions)
        action = action_queue.popleft()
        executed_from_obs = obs
        obs, _, done, _ = env.step(action.tolist())
        policy.record_history_state(executed_from_obs)
        if done:
            return True, replay_images, query_horizons
    return False, replay_images, query_horizons


def eval_libero(cfg: GenerateConfig) -> float:
    _setup_logging(cfg)
    if not cfg.ckpt_path:
        raise ValueError("--ckpt_path is required for TurboVLA evaluation.")
    if not Path(cfg.ckpt_path).exists():
        raise FileNotFoundError(f"TurboVLA checkpoint not found: {cfg.ckpt_path}")
    if not cfg.dinov3_path:
        raise ValueError("--dinov3_path is required")
    if not cfg.bert_path:
        raise ValueError("--bert_path is required")
    if cfg.r3m_path and not Path(cfg.r3m_path).is_file():
        raise FileNotFoundError(f"R3M checkpoint not found: {cfg.r3m_path}")
    if not Path(cfg.stats_path).is_file():
        raise FileNotFoundError(f"stats file not found: {cfg.stats_path}")
    if cfg.task_suite_name not in LIBERO_SUITES:
        raise ValueError(f"Unknown task suite {cfg.task_suite_name}; choose from {LIBERO_SUITES}")
    (
        PolicyClass,
        get_libero_dummy_action,
        rotate_libero_image,
        set_seed_everywhere,
    ) = _import_turbovla_adapter()
    set_seed_everywhere(cfg.seed)

    logging.info("Loading TurboVLA policy from %s", cfg.ckpt_path)
    policy = PolicyClass(
        ckpt_path=cfg.ckpt_path,
        dinov3_path=cfg.dinov3_path,
        bert_path=cfg.bert_path,
        r3m_path=cfg.r3m_path,
        stats_path=cfg.stats_path,
        stats_key=cfg.stats_key,
    )

    if cfg.dry_run_model_load:
        logging.info("dry_run_model_load=True; exiting before LIBERO rollout.")
        return 0.0

    if policy.chunk_size != cfg.chunk_size:
        raise ValueError(f"chunk_size={cfg.chunk_size} does not match checkpoint horizon={policy.chunk_size}")

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    from libero.libero import benchmark

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    n_tasks = task_suite.n_tasks
    if n_tasks != 10:
        raise ValueError("the fixed protocol requires ten tasks per suite")

    summary = {
        "script": "turbovla_libero_evaluation",
        "ckpt_path": cfg.ckpt_path,
        "task_suite_name": cfg.task_suite_name,
        **PROTOCOL,
        "eval_slice_start": 0,
        "eval_slice_stride": 1,
        "eval_episode_slice_start": cfg.eval_episode_slice_start,
        "eval_episode_slice_stride": cfg.eval_episode_slice_stride,
        "tasks": [],
        "total_episodes": 0,
        "total_successes": 0,
        "overall_success_rate": 0.0,
        "policy_queries": 0,
        "execution_horizon_counts": {},
        "mean_execution_horizon": 0.0,
    }

    total_episodes = 0
    total_successes = 0
    total_horizon_counts: Counter[int] = Counter()
    video_root = Path(cfg.video_out_path) / Path(cfg.ckpt_path).stem / cfg.task_suite_name
    if cfg.save_video:
        video_root.mkdir(parents=True, exist_ok=True)

    task_ids = list(range(n_tasks))
    if not 0 <= cfg.eval_episode_slice_start < SHARDS:
        raise ValueError("episode shard must be between 0 and 31")
    episode_ids = list(
        range(
            cfg.eval_episode_slice_start,
            cfg.num_trials_per_task,
            cfg.eval_episode_slice_stride,
        )
    )
    if not episode_ids:
        raise ValueError(
            f"episode slice start={cfg.eval_episode_slice_start} "
            f"stride={cfg.eval_episode_slice_stride} selects no episodes out of "
            f"{cfg.num_trials_per_task}"
        )
    for task_id in tqdm.tqdm(task_ids, desc="tasks"):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        if cfg.num_trials_per_task > len(initial_states):
            raise ValueError(
                f"{cfg.task_suite_name} task {task_id} has only {len(initial_states)} initial states, "
                f"but num_trials_per_task={cfg.num_trials_per_task}"
            )

        env, task_description = _make_libero_env(task, cfg)
        task_successes = 0
        task_episodes = 0
        task_horizon_counts: Counter[int] = Counter()
        try:
            for episode_idx in tqdm.tqdm(episode_ids, desc=f"task {task_id}", leave=False):
                success, replay_images, query_horizons = _run_episode(
                    cfg,
                    env,
                    policy,
                    task_description,
                    initial_states[episode_idx],
                    get_libero_dummy_action,
                    rotate_libero_image,
                )
                task_episodes += 1
                total_episodes += 1
                if success:
                    task_successes += 1
                    total_successes += 1
                task_horizon_counts.update(query_horizons)
                total_horizon_counts.update(query_horizons)

                suffix = "success" if success else "failure"
                logging.info(
                    "task=%d episode=%d success=%s running=%d/%d %.1f%%",
                    task_id,
                    episode_idx,
                    success,
                    total_successes,
                    total_episodes,
                    100.0 * total_successes / max(total_episodes, 1),
                )
                if cfg.save_video:
                    _save_video(video_root / f"task{task_id:02d}_ep{episode_idx:03d}_{suffix}.mp4", replay_images)
        finally:
            env.close()

        task_rate = task_successes / max(task_episodes, 1)
        summary["tasks"].append(
            {
                "task_id": task_id,
                "task_description": task_description,
                "episodes": task_episodes,
                "successes": task_successes,
                "success_rate": task_rate,
                "policy_queries": sum(task_horizon_counts.values()),
                "execution_horizon_counts": {
                    str(horizon): count for horizon, count in sorted(task_horizon_counts.items())
                },
                "mean_execution_horizon": (
                    sum(horizon * count for horizon, count in task_horizon_counts.items())
                    / max(sum(task_horizon_counts.values()), 1)
                ),
            }
        )
        logging.info("task=%d success_rate=%.4f (%d/%d)", task_id, task_rate, task_successes, task_episodes)

    final_rate = total_successes / max(total_episodes, 1)
    summary["total_episodes"] = total_episodes
    summary["total_successes"] = total_successes
    summary["overall_success_rate"] = final_rate
    summary["policy_queries"] = sum(total_horizon_counts.values())
    summary["execution_horizon_counts"] = {
        str(horizon): count for horizon, count in sorted(total_horizon_counts.items())
    }
    summary["mean_execution_horizon"] = (
        sum(horizon * count for horizon, count in total_horizon_counts.items())
        / max(sum(total_horizon_counts.values()), 1)
    )

    result_path = _result_path(cfg)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logging.info("Final success rate: %.4f (%d/%d)", final_rate, total_successes, total_episodes)
    logging.info("Saved result json: %s", result_path)
    return final_rate


def main():
    eval_libero(parse_args())


if __name__ == "__main__":
    main()
