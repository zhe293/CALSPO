"""A 10-second CPU-only disconnect probe. Never launches training."""
import json
import subprocess
import sys
from pathlib import Path

root=Path(__file__).resolve().parent/'detach_smoke'
root.mkdir(exist_ok=True)
result=root/'result.json'
if result.exists():
    raise SystemExit('Smoke artifact already exists; inspect it instead of overwriting')
code='import json,time,sys; time.sleep(10); open(sys.argv[1],"w").write(json.dumps({"cpu_detached_completed":True}))'
with (root/'launcher.log').open('x') as log:
    child=subprocess.Popen([sys.executable,'-c',code,str(result)],stdin=subprocess.DEVNULL,
                           stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'pid':child.pid,'result':str(result)}))
