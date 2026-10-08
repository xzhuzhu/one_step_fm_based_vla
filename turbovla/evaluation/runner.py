"""Evaluate the four LIBERO suites sequentially with 32 shards on one GPU."""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
from importlib.util import find_spec
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from .protocol import PROTOCOL, SHARDS, SUITES, TRIALS_PER_TASK


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, default=Path("one_step_fm_based_vla_95k.pth"))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation/95k_32slice"))
    parser.add_argument("--dinov3-path", type=Path, default=Path("pretrained/dinov3-vitb16"))
    parser.add_argument("--bert-path", type=Path, default=Path("pretrained/bert-base-uncased"))
    parser.add_argument("--r3m-path", type=Path, default=Path("pretrained/r3m-resnet18/backbone.pth"))
    parser.add_argument("--stats-path", type=Path, default=Path("experiments/libero/configs/libero_all4_stats.json"))
    parser.add_argument("--save-video", action="store_true")
    args = parser.parse_args()
    if args.gpu < 0:
        parser.error("--gpu must be nonnegative")
    for name in ("ckpt", "r3m_path", "stats_path"):
        path = getattr(args, name)
        if not path.is_file():
            parser.error(f"file not found: {path}")
        setattr(args, name, path.resolve())
    for name in ("dinov3_path", "bert_path"):
        path = getattr(args, name)
        if not path.is_dir():
            parser.error(f"model directory not found: {path}")
        setattr(args, name, path.resolve())
    args.output_dir = args.output_dir.resolve()
    return args


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def worker_environment(gpu: int) -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "MUJOCO_EGL_DEVICE_ID": str(gpu),
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
        "PYTHONUNBUFFERED": "1",
        "TF_CPP_MIN_LOG_LEVEL": "2",
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4",
        "NUMEXPR_NUM_THREADS": "4",
    })
    env.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", str(Path(__file__).with_name("nvidia_egl_vendor.json")))
    return env


def load_shard(path: Path, suite: str, shard: int, checkpoint: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid shard: {path}")
    if any(type(payload.get(key)) is not type(value) or payload.get(key) != value for key, value in PROTOCOL.items()):
        raise ValueError(f"protocol mismatch: {path}")
    if any(payload.get(key, False) for key in ("pace", "temporal_ensemble", "ablate_history_r3m_tokens")):
        raise ValueError(f"experimental protocol is not supported: {path}")
    if (payload.get("task_suite_name") != suite
            or payload.get("eval_episode_slice_start") != shard
            or payload.get("eval_episode_slice_stride") != SHARDS
            or payload.get("eval_slice_start", 0) != 0
            or payload.get("eval_slice_stride", 1) != 1
            or Path(payload.get("ckpt_path", "")).resolve() != checkpoint.resolve()):
        raise ValueError(f"shard identity mismatch: {path}")
    expected = len(range(shard, TRIALS_PER_TASK, SHARDS))
    tasks = payload.get("tasks", [])
    if len(tasks) != 10 or sorted(task.get("task_id", -1) for task in tasks) != list(range(10)):
        raise ValueError(f"missing or duplicate tasks: {path}")
    for task in tasks:
        if (type(task.get("episodes")) is not int or task.get("episodes") != expected
                or type(task.get("successes")) is not int
                or not 0 <= task["successes"] <= expected):
            raise ValueError(f"invalid task counts: {path}")
        queries = task.get("policy_queries", 0)
        if task.get("execution_horizon_counts") != {"10": queries} or queries <= 0:
            raise ValueError(f"invalid execution horizons: {path}")
    if (payload.get("total_episodes") != 10 * expected
            or payload.get("total_successes") != sum(task["successes"] for task in tasks)):
        raise ValueError(f"invalid shard totals: {path}")
    return payload


def aggregate_suite(paths: list[Path], suite: str, checkpoint: Path) -> dict[str, Any]:
    if len(paths) != SHARDS or len({path.resolve() for path in paths}) != SHARDS:
        raise ValueError("expected 32 distinct episode shards")
    payloads = [load_shard(path, suite, shard, checkpoint) for shard, path in enumerate(paths)]
    tasks = []
    for task_id in range(10):
        rows = [next(task for task in payload["tasks"] if task["task_id"] == task_id) for payload in payloads]
        episodes = sum(row["episodes"] for row in rows)
        if episodes != TRIALS_PER_TASK or len({row["task_description"] for row in rows}) != 1:
            raise ValueError(f"invalid coverage or instruction for task {task_id}")
        successes = sum(row["successes"] for row in rows)
        queries = sum(row["policy_queries"] for row in rows)
        tasks.append({
            "task_id": task_id,
            "task_description": rows[0]["task_description"],
            "episodes": episodes,
            "successes": successes,
            "success_rate": successes / episodes,
            "policy_queries": queries,
            "execution_horizon_counts": {"10": queries},
        })
    successes = sum(task["successes"] for task in tasks)
    return {
        **PROTOCOL,
        "task_suite_name": suite,
        "ckpt_path": str(checkpoint.resolve()),
        "shards": SHARDS,
        "tasks": tasks,
        "total_episodes": 500,
        "total_successes": successes,
        "overall_success_rate": successes / 500,
    }


def worker_command(args: argparse.Namespace, suite: str, shard: int, output: Path) -> list[str]:
    command = [
        sys.executable, "-m", "vla_adapter.rollout",
        "--ckpt_path", str(args.ckpt),
        "--dinov3_path", str(args.dinov3_path),
        "--bert_path", str(args.bert_path),
        "--r3m_path", str(args.r3m_path),
        "--stats_path", str(args.stats_path),
        "--task_suite_name", suite,
        "--eval_episode_slice_start", str(shard),
        "--result_json_path", str(output),
        "--video_out_path", str(args.output_dir / "videos"),
    ]
    if args.save_video:
        command.append("--save_video")
    return command


def evaluate_suite(args: argparse.Namespace, suite: str) -> dict[str, Any]:
    directory = args.output_dir / suite
    directory.mkdir(parents=True, exist_ok=True)
    paths = [directory / f"slice{shard}-of-{SHARDS}.json" for shard in range(SHARDS)]
    jobs = []
    try:
        for shard, path in enumerate(paths):
            try:
                load_shard(path, suite, shard, args.ckpt)
                continue
            except (OSError, ValueError, KeyError, TypeError):
                pass
            log = path.with_suffix(".log").open("w", encoding="utf-8")
            try:
                process = subprocess.Popen(
                    worker_command(args, suite, shard, path),
                    env=worker_environment(args.gpu), stdout=log, stderr=subprocess.STDOUT,
                )
            except BaseException:
                log.close()
                raise
            jobs.append((shard, process, log))
            print(f"{suite}: started shard {shard + 1}/{SHARDS} on GPU {args.gpu}", flush=True)
        for shard, process, _ in jobs:
            if process.wait() != 0:
                raise RuntimeError(f"{suite} shard {shard} failed; inspect {paths[shard].with_suffix('.log')}")
        result = aggregate_suite(paths, suite, args.ckpt)
        write_json(directory / "summary.json", result)
        print(f"{suite}: {result['total_successes']}/500 ({result['overall_success_rate']:.2%})", flush=True)
        return result
    finally:
        for _, process, log in jobs:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            log.close()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    def fingerprint(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    checkpoint_sha256 = fingerprint(args.ckpt)
    inputs = {"stats": fingerprint(args.stats_path), "r3m": fingerprint(args.r3m_path)}
    for name in ("dinov3_path", "bert_path"):
        for path in sorted(getattr(args, name).iterdir()):
            if path.is_file() and path.suffix in {".json", ".txt", ".safetensors"}:
                inputs[f"{name}/{path.name}"] = fingerprint(path)
    versions = {name: version(name) for name in ("torch", "transformers", "numpy", "robosuite", "mujoco", "triton")}
    source = hashlib.sha256()
    worker_package = find_spec("vla_adapter")
    if worker_package is None or worker_package.origin is None:
        raise RuntimeError("vla_adapter worker package is not installed")
    for package in (Path(__file__).resolve().parents[1], Path(worker_package.origin).parent):
        for path in sorted(package.rglob("*.py")):
            source.update(str(path.relative_to(package)).encode())
            source.update(path.read_bytes())
    source_sha256 = source.hexdigest()
    manifest_path = args.output_dir / "run_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (previous.get("checkpoint_sha256") != checkpoint_sha256
                or previous.get("inputs") != inputs or previous.get("versions") != versions
                or previous.get("source_sha256") != source_sha256
                or previous.get("gpu") != args.gpu):
            raise ValueError("output directory belongs to a different run configuration; use another --output-dir")
    write_json(manifest_path, {
        **PROTOCOL, "gpu": args.gpu, "shards": SHARDS, "suites": list(SUITES),
        "checkpoint": str(args.ckpt), "checkpoint_sha256": checkpoint_sha256,
        "inputs": inputs, "versions": versions,
        "source_sha256": source_sha256,
    })
    results = {suite: evaluate_suite(args, suite) for suite in SUITES}
    successes = sum(result["total_successes"] for result in results.values())
    summary = {
        **PROTOCOL, "gpu": args.gpu, "shards": SHARDS, "checkpoint_sha256": checkpoint_sha256,
        "suites": {suite: {
            "episodes": result["total_episodes"], "successes": result["total_successes"],
            "success_rate": result["overall_success_rate"],
        } for suite, result in results.items()},
        "total_episodes": 2000, "total_successes": successes, "overall_success_rate": successes / 2000,
    }
    write_json(args.output_dir / "summary_all4.json", summary)
    print(f"All four: {successes}/2000 ({successes / 2000:.2%})", flush=True)


if __name__ == "__main__":
    main()
