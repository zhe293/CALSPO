"""Run the two approved HARPO conditions after the completed GRPO baseline.

Use a new output root so the failed, zero-update HARPO attempt stays intact.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--baseline-dir', type=Path, required=True)
    parser.add_argument('--detach', action='store_true')
    args = parser.parse_args()
    root = args.output_root.resolve(strict=True)
    baseline = args.baseline_dir.resolve(strict=True)
    owned_prefix = '/data/codefile/changshaoqing/harpo_p40_runs/'
    if not str(root).startswith(owned_prefix) or not str(baseline).startswith(owned_prefix):
        raise ValueError('Output and baseline must be in the personal run directory')
    if root.stat().st_uid != os.getuid():
        raise PermissionError('Output root is not owned by the current user')
    here = Path(__file__).resolve().parent
    if args.detach:
        with (root/'suite.log').open('x') as log:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                                      '--output-root', str(root), '--baseline-dir', str(baseline)],
                                     stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        (root/'suite.pid').write_text(str(child.pid))
        print(f'harpo_pair_pid={child.pid}')
        return

    state_path = root/'suite_status.json'
    def state(value, **extra):
        temp = root/'suite_status.tmp'
        temp.write_text(json.dumps({'state': value, 'time': time.time(), **extra}, indent=2))
        temp.replace(state_path)

    try:
        manifest = json.loads((baseline/'run_manifest.json').read_text())
        baseline_status = json.loads((baseline/'status.json').read_text())
        if manifest['run'] != 'grpo_reward_a' or manifest['seed'] != 45 or manifest['total_steps'] != 16:
            raise ValueError('Unexpected baseline design')
        if baseline_status['state'] != 'complete':
            raise ValueError('GRPO baseline has not passed')
        subprocess.run([sys.executable, str(here/'verify_run_complete.py'),
                        '--run-dir', str(baseline), '--expected-steps', '16'], check=True)
        if len(manifest['gpus']) != 1 or not manifest['gpus'][0].isdigit():
            raise ValueError('Expected one explicit GPU')
        gpu = manifest['gpus'][0]
        deadline = time.time()+5*24*3600-600
        reference_hashes = None
        for run in ('harpo_reward_a', 'harpo_reward_b'):
            if time.time() >= deadline:
                raise TimeoutError('HARPO pair deadline reached')
            if reference_hashes is not None:
                for name, digest in reference_hashes.items():
                    if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
                        raise RuntimeError(f'Code/data drift between HARPO runs: {name}')
            command = [sys.executable, str(here/'run_experiment.py'), '--run', run,
                       '--repo', str(here.parents[1]), '--model', manifest['model'],
                       '--reward-model', manifest['reward_model'],
                       '--train', manifest['train'], '--validation', manifest['validation'],
                       '--output-root', str(root), '--gpus', gpu,
                       '--total-steps', '16', '--seed', '45',
                       '--execute', '--registration-confirmed']
            state('running', run=run)
            with (root/(run+'.launcher.log')).open('x') as log:
                proc = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                        stdout=log, stderr=subprocess.STDOUT,
                                        start_new_session=True)
                (root/(run+'.pid')).write_text(str(proc.pid))
                while proc.poll() is None:
                    if time.time() >= deadline:
                        launcher = psutil.Process(proc.pid)
                        monitors = [child for child in launcher.children(recursive=True)
                                    if child.uids().real == os.getuid()
                                    and any(arg.endswith('/resource_monitor.py') for arg in child.cmdline())]
                        for monitor in monitors:
                            monitor.terminate()
                        psutil.wait_procs(monitors, timeout=30)
                        raise TimeoutError('HARPO pair deadline reached with run active')
                    time.sleep(60)
                if proc.returncode:
                    raise RuntimeError(f'{run} exited {proc.returncode}; queue stopped')
            status = json.loads((root/run/'status.json').read_text())
            if status['state'] != 'complete':
                raise RuntimeError(f'{run} did not pass verification')
            current_manifest = json.loads((root/run/'run_manifest.json').read_text())
            if any(current_manifest[key] != manifest[key] for key in ('model','reward_model','train','validation','seed','total_steps','gpus')):
                raise RuntimeError(f'{run} differs from baseline input/config identity')
            reference_hashes = current_manifest['sha256']
        state('complete', message='Both HARPO runs passed; update registration')
    except BaseException as exc:
        state('failed', error=str(exc))
        raise


if __name__ == '__main__':
    main()
