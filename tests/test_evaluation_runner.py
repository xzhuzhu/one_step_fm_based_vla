import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from turbovla.evaluation import runner
from turbovla.evaluation.protocol import PROTOCOL, SHARDS, TRIALS_PER_TASK


def shard_payload(checkpoint, shard):
    episodes = len(range(shard, TRIALS_PER_TASK, SHARDS))
    return {
        **PROTOCOL,
        "ckpt_path": str(checkpoint),
        "task_suite_name": "libero_spatial",
        "eval_episode_slice_start": shard,
        "eval_episode_slice_stride": SHARDS,
        "tasks": [{
            "task_id": task,
            "task_description": f"instruction {task}",
            "episodes": episodes,
            "successes": episodes,
            "policy_queries": episodes,
            "execution_horizon_counts": {"10": episodes},
        } for task in range(10)],
        "total_episodes": episodes * 10,
        "total_successes": episodes * 10,
    }


def make_shards(tmp_path):
    checkpoint = tmp_path / "one_step_fm_based_vla_95k.pth"
    checkpoint.write_bytes(b"checkpoint")
    paths = []
    for shard in range(SHARDS):
        path = tmp_path / f"slice{shard}-of-{SHARDS}.json"
        path.write_text(json.dumps(shard_payload(checkpoint, shard)))
        paths.append(path)
    return checkpoint, paths


def test_aggregate_verifies_uneven_shards_cover_fifty_initial_states(tmp_path):
    checkpoint, paths = make_shards(tmp_path)
    result = runner.aggregate_suite(paths, "libero_spatial", checkpoint)
    assert result["total_episodes"] == 500
    assert result["total_successes"] == 500
    assert all(task["episodes"] == 50 for task in result["tasks"])


@pytest.mark.parametrize("mutation", [
    lambda p: p.update(seed=43),
    lambda p: p.update(seed=42.0),
    lambda p: p.update(precision="fp32"),
    lambda p: p.update(temporal_ensemble=True),
    lambda p: p.update(eval_episode_slice_start=1),
    lambda p: p["tasks"].__setitem__(1, p["tasks"][0]),
    lambda p: p["tasks"][0].update(episodes=1),
    lambda p: p["tasks"][0].update(successes=3),
    lambda p: p["tasks"][0].update(execution_horizon_counts={"12": 2}),
    lambda p: p.update(total_successes=0),
])
def test_stale_incomplete_or_malformed_shards_are_rejected(tmp_path, mutation):
    checkpoint, paths = make_shards(tmp_path)
    payload = json.loads(paths[0].read_text())
    mutation(payload)
    paths[0].write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        runner.aggregate_suite(paths, "libero_spatial", checkpoint)


def test_duplicate_or_missing_shards_are_rejected(tmp_path):
    checkpoint, paths = make_shards(tmp_path)
    with pytest.raises(ValueError, match="32 distinct"):
        runner.aggregate_suite(paths[:-1], "libero_spatial", checkpoint)
    with pytest.raises(ValueError, match="32 distinct"):
        runner.aggregate_suite([paths[0]] * 32, "libero_spatial", checkpoint)


def test_completed_shards_resume_without_spawning_workers(tmp_path, monkeypatch):
    checkpoint, paths = make_shards(tmp_path)
    directory = tmp_path / "libero_spatial"
    directory.mkdir()
    for path in paths:
        path.rename(directory / path.name)
    spawn = Mock(side_effect=AssertionError("completed shards must not run again"))
    monkeypatch.setattr(runner.subprocess, "Popen", spawn)
    result = runner.evaluate_suite(SimpleNamespace(output_dir=tmp_path, ckpt=checkpoint), "libero_spatial")
    assert result["total_episodes"] == 500
    spawn.assert_not_called()


def test_all_thirty_two_workers_share_one_gpu(tmp_path, monkeypatch):
    checkpoint = tmp_path / "one_step_fm_based_vla_95k.pth"
    checkpoint.write_bytes(b"checkpoint")
    args = SimpleNamespace(
        output_dir=tmp_path, ckpt=checkpoint, gpu=2, save_video=False,
        dinov3_path=tmp_path, bert_path=tmp_path, r3m_path=tmp_path, stats_path=tmp_path,
    )
    environments = []

    def spawn(command, **kwargs):
        shard = int(command[command.index("--eval_episode_slice_start") + 1])
        output = Path(command[command.index("--result_json_path") + 1])
        output.write_text(json.dumps(shard_payload(checkpoint, shard)))
        environments.append(kwargs["env"])
        return Mock(wait=Mock(return_value=0), poll=Mock(return_value=0))

    monkeypatch.setattr(runner.subprocess, "Popen", spawn)
    result = runner.evaluate_suite(args, "libero_spatial")
    assert len(environments) == 32
    assert all(env["CUDA_VISIBLE_DEVICES"] == "2" for env in environments)
    assert all(env["MUJOCO_EGL_DEVICE_ID"] == "2" for env in environments)
    assert result["total_episodes"] == 500


def test_launch_failure_terminates_existing_workers(tmp_path, monkeypatch):
    checkpoint = tmp_path / "one_step_fm_based_vla_95k.pth"
    checkpoint.write_bytes(b"checkpoint")
    args = SimpleNamespace(
        output_dir=tmp_path, ckpt=checkpoint, gpu=2, save_video=False,
        dinov3_path=tmp_path, bert_path=tmp_path, r3m_path=tmp_path, stats_path=tmp_path,
    )
    process = Mock(poll=Mock(return_value=None), wait=Mock(return_value=0))
    monkeypatch.setattr(runner.subprocess, "Popen", Mock(side_effect=[process, OSError("launch failed")]))
    with pytest.raises(OSError, match="launch failed"):
        runner.evaluate_suite(args, "libero_spatial")
    process.terminate.assert_called_once()


def test_worker_cli_rejects_retired_protocol_flags(monkeypatch):
    from vla_adapter.rollout import parse_args
    monkeypatch.setattr("sys.argv", ["worker", "--precision", "fp32"])
    with pytest.raises(SystemExit):
        parse_args()


@pytest.mark.parametrize("changed", ["checkpoint", "statistics", "versions", "gpu"])
def test_resume_manifest_rejects_changed_run_inputs(tmp_path, monkeypatch, changed):
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"weights")
    statistics = tmp_path / "stats.json"
    statistics.write_bytes(b"statistics")
    r3m = tmp_path / "r3m.pth"
    r3m.write_bytes(b"r3m")
    args = SimpleNamespace(
        ckpt=checkpoint, stats_path=statistics, r3m_path=r3m,
        dinov3_path=tmp_path / "dino", bert_path=tmp_path / "bert",
        output_dir=tmp_path / "output", gpu=2,
    )
    args.dinov3_path.mkdir()
    args.bert_path.mkdir()
    monkeypatch.setattr(runner, "parse_args", lambda: args)
    monkeypatch.setattr(runner, "version", lambda _: "initial")
    monkeypatch.setattr(runner, "evaluate_suite", lambda *_: {
        "total_episodes": 500, "total_successes": 500, "overall_success_rate": 1.0,
    })
    runner.main()
    if changed == "checkpoint":
        checkpoint.write_bytes(b"changed weights")
    elif changed == "statistics":
        statistics.write_bytes(b"changed statistics")
    elif changed == "versions":
        monkeypatch.setattr(runner, "version", lambda _: "changed")
    else:
        args.gpu = 3
    with pytest.raises(ValueError, match="different run configuration"):
        runner.main()
