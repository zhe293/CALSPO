"""Verify that one controlled run produced training, resource, and final-test artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


EXPECTED_SEED = 45


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-steps", type=int, required=True)
    args = parser.parse_args()

    manifest_path = args.run_dir / "run_manifest.json"
    summary_path = args.run_dir / "resource_usage.summary.json"
    csv_path = args.run_dir / "resource_usage.csv"
    log_path = args.run_dir / "training.log"
    latest_path = args.run_dir / "latest_checkpointed_iteration.txt"
    test_dir = args.run_dir / "final_test"

    for path in (manifest_path, summary_path, csv_path, log_path, latest_path):
        require(path.is_file() and path.stat().st_size > 0, f"missing or empty artifact: {path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    require(manifest.get("seed") == EXPECTED_SEED, f"seed is not {EXPECTED_SEED}: {manifest.get('seed')}")
    require(manifest.get("total_steps") == args.expected_steps, "manifest total_steps mismatch")
    require(summary.get("return_code") == 0, f"training return code: {summary.get('return_code')}")

    completed_step = int(latest_path.read_text(encoding="utf-8").strip())
    require(completed_step == args.expected_steps, f"checkpoint step {completed_step} != {args.expected_steps}")
    actor_dir = args.run_dir / f"global_step_{completed_step}" / "actor"
    require(actor_dir.is_dir(), f"missing final actor checkpoint: {actor_dir}")
    world_size = len(manifest['gpus'])
    for rank in range(world_size):
        shard = actor_dir / f'model_world_size_{world_size}_rank_{rank}.pt'
        require(shard.is_file() and shard.stat().st_size > 0, f'missing checkpoint shard: {shard}')
    reload_marker = json.loads((args.run_dir/'checkpoint_reloaded.json').read_text())
    require(reload_marker['step'] == completed_step, 'final weights were not explicitly reloaded')

    require(test_dir.is_dir(), f"missing final test directory: {test_dir}")
    test_files = [path for path in test_dir.rglob("*") if path.is_file() and path.stat().st_size > 0]
    require(test_files, f"no final test artifacts under: {test_dir}")
    predictions=json.loads((test_dir/f'predictions_step_{completed_step}.json').read_text())
    metrics=json.loads((test_dir/f'metrics_step_{completed_step}.json').read_text())
    require(len(predictions)==manifest['validation_rows']==metrics['count'], 'incomplete evaluation coverage')
    import pyarrow.parquet as pq
    truth=pq.read_table(manifest['validation'],columns=['task','answer']).to_pylist()
    require([(p['task'],p['label']) for p in predictions] == [(r['task'],r['answer'].strip().lower()) for r in truth],
            'evaluation order/labels differ from frozen data')
    for task,values in metrics['per_task'].items():
        for key in ('accuracy','macro_f1','positive_recall','balanced_accuracy','invalid_rate'):
            require(math.isfinite(values[key]) and 0<=values[key]<=1, f'invalid {task}/{key}')
    require(summary.get('monitor_error') is None, 'resource monitor failed')
    require(set(summary['gpus'])==set(manifest['gpus']), 'GPU telemetry coverage mismatch')
    steps=[json.loads(line) for line in (args.run_dir/'steps.jsonl').read_text().splitlines() if line]
    require([r['step'] for r in steps]==list(range(1,completed_step+1)), 'missing or duplicated step records')
    require(all(r['responses']==32 and r['response_tokens']>0 for r in steps), 'rollout counts/tokens mismatch')
    tokens=sum(r['response_tokens'] for r in steps)
    training=summary['phases'].get('train',{})
    require(training.get('wall_time_s',0)>0, 'missing training phase measurements')
    comparison={'steps':completed_step,'response_tokens':tokens,'train_wall_time_s':training['wall_time_s'],
                'train_seconds_per_step':training['wall_time_s']/completed_step,
                'train_seconds_per_response_token':training['wall_time_s']/tokens,
                'train_gpu_hours':training['gpu_hours'],'train_estimated_energy_wh':training['energy_wh'],
                'end_to_end_wall_time_s':summary['wall_time_s'],'end_to_end_gpu_hours':summary['gpu_hours']}
    (args.run_dir/'comparison_resources.json').write_text(json.dumps(comparison,indent=2),encoding='utf-8')

    print(f"run={manifest['run']}")
    print(f"seed={manifest['seed']}")
    print(f"completed_steps={completed_step}")
    print(f"wall_time_s={summary['wall_time_s']}")
    print(f"total_gpu_energy_wh={summary['total_gpu_energy_wh']}")
    print(f"final_test_files={len(test_files)}")
    print("run_complete=PASS")


if __name__ == "__main__":
    main()
