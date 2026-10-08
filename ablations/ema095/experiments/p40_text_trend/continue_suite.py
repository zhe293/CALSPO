"""Continue the approved three-condition suite after GRPO-A passes verification.

Run detached explicitly. Failure stops the queue; each launcher rechecks GPU freedom.
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
    parser.add_argument('--detach', action='store_true')
    args = parser.parse_args()
    root = args.output_root.resolve(strict=True)
    assert str(root).startswith('/data/codefile/changshaoqing/harpo_p40_runs/')
    assert root.stat().st_uid == os.getuid()
    here = Path(__file__).resolve().parent
    if args.detach:
        with (root/'suite.log').open('x') as log:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--output-root', str(root)],
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        (root/'suite.pid').write_text(str(child.pid))
        print(f'suite_pid={child.pid}')
        return
    state_path = root/'suite_status.json'
    def state(value, **extra):
        temp = root/'suite_status.tmp'
        temp.write_text(json.dumps({'state':value,'time':time.time(),**extra},indent=2))
        temp.replace(state_path)
    first = root/'grpo_reward_a'
    manifest = json.loads((first/'run_manifest.json').read_text())
    assert manifest['seed']==45 and manifest['total_steps']==16
    assert len(manifest['gpus'])==1 and manifest['gpus'][0].isdigit()
    selected_gpu=manifest['gpus'][0]
    deadline = time.time()+5*24*3600-600
    owned_launcher = None
    def timeout_guard():
        if time.time() < deadline:
            return
        # Only stop a verified launcher and its own monitor descendants.
        if owned_launcher is not None and owned_launcher.is_running():
            children = owned_launcher.children(recursive=True)
            monitors = [p for p in children if p.uids().real==os.getuid()
                        and any(x.endswith('/resource_monitor.py') for x in p.cmdline())]
            for process in monitors:
                process.terminate()
            psutil.wait_procs(monitors,timeout=30)
        raise TimeoutError('Suite deadline before five days reached; queue stopped')
    try:
        initial = json.loads((first/'status.json').read_text())
        if initial['state']=='running':
            candidate = psutil.Process(initial['pid'])
            assert candidate.uids().real==os.getuid()
            assert str(here/'run_experiment.py') in candidate.cmdline()
            owned_launcher = candidate
        state('waiting_for_grpo')
        while True:
            timeout_guard()
            status = json.loads((first/'status.json').read_text())
            if status['state']=='complete':
                break
            if status['state']=='failed':
                raise RuntimeError('GRPO-A failed; no HARPO run launched')
            if owned_launcher is None or not owned_launcher.is_running():
                raise RuntimeError('GRPO-A launcher disappeared without completion')
            time.sleep(60)
        # Recheck first result and freeze code/data equality before advancing.
        subprocess.run([sys.executable,str(here/'verify_run_complete.py'),'--run-dir',str(first),
                        '--expected-steps','16'],check=True)
        for run in ('harpo_reward_a','harpo_reward_b'):
            timeout_guard()
            for name,digest in manifest['sha256'].items():
                if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:
                    raise RuntimeError(f'Input/code changed since GRPO-A: {name}')
            command=[sys.executable,str(here/'run_experiment.py'),'--run',run,
                     '--repo',str(here.parents[1]),'--model',manifest['model'],
                     '--reward-model',manifest['reward_model'],'--train',manifest['train'],
                     '--validation',manifest['validation'],'--output-root',str(root),
                     '--gpus',selected_gpu,'--total-steps','16','--seed','45','--execute','--registration-confirmed']
            state('running',run=run)
            with (root/(run+'.launcher.log')).open('x') as log:
                process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,
                                         stderr=subprocess.STDOUT,start_new_session=True)
                owned_launcher=psutil.Process(process.pid)
                (root/(run+'.pid')).write_text(str(process.pid))
                while process.poll() is None:
                    timeout_guard()
                    time.sleep(60)
                if process.returncode:
                    raise RuntimeError(f'{run} exited {process.returncode}; queue stopped')
            result=json.loads((root/run/'status.json').read_text())
            if result['state']!='complete':
                raise RuntimeError(f'{run} did not pass verification')
        state('complete',message='All three runs passed; update resource registration')
    except BaseException as exc:
        state('failed',error=str(exc))
        raise


if __name__=='__main__':
    main()
