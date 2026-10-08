"""CPU-side progress only: no tensor values, synchronization or RNG changes."""
import json
import os
from pathlib import Path
import time


def event(name, **values):
    root = os.environ.get('HBA_RUN_DIR')
    if not root:
        return
    record = {'time':time.time(), 'pid':os.getpid(), 'event':name, **values}
    phase = Path(root)/'phase.json'
    if phase.exists():
        record['step'] = json.loads(phase.read_text())['step']
    line = json.dumps(record)
    with (Path(root)/f'progress_{os.getpid()}.jsonl').open('a') as stream:
        stream.write(line+'\n')
    print('[HBA_PROGRESS] '+line, flush=True)


def token_observer(prompt_length):
    from transformers import StoppingCriteria, StoppingCriteriaList
    class Observer(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            count = input_ids.shape[-1]-prompt_length
            if count == 1 or count % 32 == 0:
                event('generation_tokens', tokens=count)
            return False  # observation never stops generation
    return StoppingCriteriaList([Observer()])
