"""CPU-only proof that a failed first run never launches the next conditions."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

with tempfile.TemporaryDirectory(prefix='suite_guard_',dir='/data/codefile/anonymous/harpo_p40_runs') as tmp:
    root=Path(tmp)
    first=root/'grpo_reward_a'
    first.mkdir()
    (first/'run_manifest.json').write_text(json.dumps({'seed':45,'total_steps':16,'gpus':['0']}))
    (first/'status.json').write_text(json.dumps({'state':'failed'}))
    result=subprocess.run([sys.executable,str(Path(__file__).with_name('continue_suite.py')),
                           '--output-root',str(root)],capture_output=True,text=True,timeout=20)
    assert result.returncode != 0
    assert json.loads((root/'suite_status.json').read_text())['state']=='failed'
    assert not (root/'harpo_reward_a').exists() and not (root/'harpo_reward_b').exists()
    assert not (root/'harpo_reward_a.launcher.log').exists()
print('suite_failure_guard=PASS (CPU only; no training launched)')
