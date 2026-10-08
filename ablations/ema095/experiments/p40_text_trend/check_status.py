"""Print a compact status snapshot without touching GPU jobs."""
import argparse
import json
import time
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run-dir',type=Path,required=True)
    args=ap.parse_args()
    out={'run':args.run_dir.name,'checked_at':time.time()}
    for name in ('status','phase'):
        path=args.run_dir/(name+'.json')
        out[name]=json.loads(path.read_text()) if path.exists() else 'not_created'
    steps=args.run_dir/'steps.jsonl'
    if steps.exists():
        lines=steps.read_text().splitlines()
        if lines:
            record=json.loads(lines[-1])
            out['last_completed_step']=record['step']
            out['seconds_since_completed_step']=time.time()-record['time']
    log=args.run_dir/'training.log'
    if log.exists():
        out['seconds_since_log_update']=time.time()-log.stat().st_mtime
    telemetry=args.run_dir/'resource_usage.csv'
    if telemetry.exists():
        out['seconds_since_telemetry_update']=time.time()-telemetry.stat().st_mtime
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
