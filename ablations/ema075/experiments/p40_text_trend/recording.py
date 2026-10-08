"""Driver-side phase and step records, flushed throughout the run."""
import json
import os
import time
from pathlib import Path

def phase(name,step=0):
    root=os.environ.get('HBA_RUN_DIR')
    if not root:
        return
    record={'time':time.time(),'phase':name,'step':int(step)}
    path=Path(root)
    with (path/'phases.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(record)+'\n')
        f.flush()
    temp=path/'phase.tmp'
    temp.write_text(json.dumps(record),encoding='utf-8')
    temp.replace(path/'phase.json')

def record_step(metrics,batch,step):
    root=os.environ.get('HBA_RUN_DIR')
    if not root:
        return
    record={'step':int(step),'time':time.time(),'response_tokens':int(batch.batch['response_mask'].sum().item()),
            'responses':len(batch),'metrics':metrics}
    with (Path(root)/'steps.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(record,default=lambda x:x.item() if hasattr(x,'item') else str(x))+'\n')
        f.flush()
