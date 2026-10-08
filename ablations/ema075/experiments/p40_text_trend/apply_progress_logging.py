"""Add per-answer/token and forward/backward progress; keep computation unchanged."""
import argparse
import ast
import hashlib
from pathlib import Path

ap=argparse.ArgumentParser()
ap.add_argument('--repo',type=Path,required=True)
args=ap.parse_args()
root=args.repo/'verl/verl'
changes={}
def patch(rel, old, new):
    path=root/rel
    source=changes.get(path,path.read_text(encoding='utf-8'))
    if new in source:
        return
    assert source.count(old)==1, (rel,old)
    changes[path]=source.replace(old,new)

hf='workers/rollout/hf_rollout.py'
actor='workers/actor/dp_actor.py'
patch(hf,'import contextlib\n','import contextlib\nimport time\nfrom live_progress import event, token_observer\n')
patch(hf,'        output = [self._generate_minibatch(p) for p in batch_prompts]\n',
'''        output = []
        for number, prompt in enumerate(batch_prompts, 1):
            started = time.monotonic()
            event('answer_start', answer=number, total=len(batch_prompts))
            output.append(self._generate_minibatch(prompt))
            event('answer_end', answer=number, total=len(batch_prompts), elapsed_s=time.monotonic()-started)
''')
patch(hf,'                generation_config=generation_config,\n',
      '                generation_config=generation_config,\n                stopping_criteria=token_observer(prompt_length),\n')
patch(actor,'import logging\n','import logging\nimport time\nfrom live_progress import event\n')
patch(actor,'\n        for micro_batch in micro_batches:\n',
'''
        for number, micro_batch in enumerate(micro_batches, 1):
            event('logprob_start', microbatch=number, total=len(micro_batches))
''')
patch(actor,'            log_probs_lst.append(log_probs)\n',
      "            event('logprob_end', microbatch=number, total=len(micro_batches))\n            log_probs_lst.append(log_probs)\n")
patch(actor,'\n                for micro_batch in micro_batches:\n',
'''
                for number, micro_batch in enumerate(micro_batches, 1):
                    started = time.monotonic()
                    event('update_forward_start', microbatch=number, total=len(micro_batches))
''')
patch(actor,'                    self._p40_scaler.scale(loss).backward()\n',
'''                    event('backward_start', microbatch=number)
                    self._p40_scaler.scale(loss).backward()
                    event('backward_end', microbatch=number, forward_backward_elapsed_s=time.monotonic()-started)
''')
patch(actor,'                grad_norm = self._optimizer_step()\n',
'''                event('optimizer_start')
                grad_norm = self._optimizer_step()
                event('optimizer_end')
''')
for path,source in changes.items():
    ast.parse(source)
backup=args.repo/'experiments/p40_text_trend/patch_backups'
backup.mkdir(exist_ok=True)
for path,source in changes.items():
    original=path.read_bytes()
    saved=backup/(path.name+'.'+hashlib.sha256(original).hexdigest()+'.bak')
    if not saved.exists():
        saved.write_bytes(original)
    path.write_text(source,encoding='utf-8')
print('progress_logging=PASS files='+str(len(changes)))
