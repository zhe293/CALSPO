"""Start the approved 304-row comparison after both existing jobs release GPUs."""

import hashlib
import json
import os
import subprocess
import sys
import time
import fcntl
from pathlib import Path


REPO = Path("/data/codefile/changshaoqing/human_behavior_atlas_xrpo_304")
EXPERIMENT = REPO / "experiments/xrpo_multimodal_304"
DATASET = Path("/data/datasets/human_behavior_atlas-data/changshaoqing_multimodal_304_v4")
OUTPUT = Path("/data/codefile/changshaoqing/xrpo_multimodal_304_results_20260924")
MODEL = Path("/share/155/pretrained/Qwen2.5-Omni-3B")
REWARD_MODEL = Path("/share/155/pretrained/mashang/SAGEPO/reward_models/all-MiniLM-L6-v2")
OLD_JOBS = (
    Path("/data/codefile/changshaoqing/harpo_p40_runs/harpo_b_gpu2_repeat_20260924/harpo_reward_b"),
    Path("/data/codefile/changshaoqing/harpo_p40_runs/grpo_b_gpu0_20260924/grpo_reward_b"),
)
TARGETS = (("original_harpo", 0), ("xrpo_23", 2))
PINNED_FILES = (
    DATASET / "train.parquet", DATASET / "validation.parquet", DATASET / "manifest.json",
    EXPERIMENT / "run_multimodal_304.py", EXPERIMENT / "reward_b_local.py",
    EXPERIMENT / "original_reward.py", EXPERIMENT / "multimodal_evaluation.py",
    EXPERIMENT / "omni_positions.py",
    EXPERIMENT / "cpu_audit_result.json", EXPERIMENT / "verify_multimodal_304.py",
    REPO / "experiments/p40_text_trend/resource_monitor.py",
    REPO / "verl/verl/workers/actor/dp_actor.py",
    REPO / "verl/verl/workers/fsdp_workers.py",
    REPO / "verl/verl/trainer/ppo/harpo_utils.py",
    REPO / "verl/verl/trainer/main_ppo.py",
    REPO / "verl/verl/trainer/ppo/ray_trainer.py",
    REPO / "verl/verl/trainer/ppo/xrpo_advantage.py",
    REPO / "verl/verl/trainer/ppo/core_algos.py",
    REPO / "verl/verl/workers/rollout/hf_rollout.py",
    REPO / "verl/verl/workers/reward_manager/batch.py",
    REPO / "verl/verl/utils/dataset/rl_dataset.py",
    REPO / "verl/verl/utils/dataset/audio_utils.py",
    REPO / "verl/verl/utils/dataset/vision_utils.py",
)


def source_hashes():
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in PINNED_FILES}


def write_status(**values):
    (OUTPUT / "watcher_status.json").write_text(json.dumps({"time": time.time(), **values}, indent=2))


def old_jobs_complete():
    for root in OLD_JOBS:
        status_file = root / "status.json"
        if not status_file.exists():
            return False
        status = json.loads(status_file.read_text())
        if status.get("state") == "failed":
            raise RuntimeError(f"Earlier job failed: {root}: {status}")
        if status.get("state") != "complete":
            return False
    return True


def main():
    if os.getuid() != REPO.stat().st_uid:
        raise PermissionError("Code copy is not owned by the current account")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock = (OUTPUT / "watcher.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    previous_launches = []
    if (OUTPUT / "launch_manifest.json").exists():
        previous_launches = json.loads((OUTPUT / "launch_manifest.json").read_text()).get("jobs", [])
        if any((OUTPUT / item["run"]).exists() for item in previous_launches):
            raise FileExistsError("An earlier study run has an output; refusing duplicate launch")
    pinned = source_hashes()
    (OUTPUT / "watcher_source_lock.json").write_text(json.dumps(pinned, indent=2))
    sys.path.insert(0, str(REPO / "experiments/p40_text_trend"))
    from resource_monitor import assert_free_gpus

    pending = {0: ("original_harpo", OLD_JOBS[1]), 2: ("xrpo_23", OLD_JOBS[0])}
    children, handles, launched, results = {}, [], previous_launches, {}
    try:
        while pending or children:
            for gpu, (run, old_root) in list(pending.items()):
                old_status = json.loads((old_root / "status.json").read_text())
                if old_status.get("state") == "failed":
                    raise RuntimeError(f"Earlier job failed: {old_root}")
                if old_status.get("state") != "complete":
                    continue
                try:
                    assert_free_gpus({gpu})
                except RuntimeError:
                    continue
                if source_hashes() != pinned:
                    raise RuntimeError("Code or data changed while waiting; refusing launch")
                if (OUTPUT / run).exists():
                    raise FileExistsError(f"Existing output for {run}")
                command = [
                    sys.executable, str(EXPERIMENT / "run_multimodal_304.py"),
                    "--run", run, "--repo", str(REPO), "--model", str(MODEL),
                    "--reward-model", str(REWARD_MODEL), "--dataset", str(DATASET),
                    "--output-root", str(OUTPUT), "--gpu", str(gpu),
                    "--execute", "--registration-confirmed",
                ]
                launched.append({"run":run,"gpu":gpu,"time":time.time(),"command":command})
                (OUTPUT / "launch_manifest.json").write_text(json.dumps({
                    "registration_confirmed_by_user":True,"jobs":launched},indent=2))
                log=(OUTPUT / f"launcher_{run}.log").open("w",buffering=1)
                handles.append(log)
                child=subprocess.Popen(command,cwd=REPO,stdin=subprocess.DEVNULL,
                                       stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                children[run]=(gpu,child)
                del pending[gpu]
            for run,(gpu,child) in list(children.items()):
                if child.poll() is not None:
                    results[run]=child.returncode
                    del children[run]
            write_status(state="running_or_waiting" if children else "waiting_for_existing_jobs",
                         waiting_gpus=list(pending),
                         jobs=[{"run":run,"gpu":gpu,"pid":child.pid} for run,(gpu,child) in children.items()],
                         return_codes=results)
            if pending or children:
                time.sleep(30)
        write_status(state="complete" if all(code==0 for code in results.values()) else "failed",
                     return_codes=results)
    finally:
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        if OUTPUT.is_dir():
            write_status(state="failed", error=str(exc))
        raise
