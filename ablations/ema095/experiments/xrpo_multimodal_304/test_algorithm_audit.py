"""Execute original and patched HARPO functions with their real repository helpers."""
import ast
import copy
import math
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from verl.trainer.ppo import harpo_utils as h

ROOT = Path(__file__).resolve().parent

def load_function(path):
    tree = ast.parse(path.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'compute_harpo_outcome_advantage')
    fn.decorator_list = []
    ns = dict(vars(h), np=np, torch=torch, math=math, defaultdict=defaultdict, EPS_DEFAULT=1e-6)
    code = 'from __future__ import annotations\n' + ast.unparse(fn)
    exec(compile(code, str(path), 'exec'), ns)
    return ns[fn.name]

def reset():
    h.task_stats.clear()
    h.dataset_stats.clear()

def args(rewards, single=False):
    return dict(token_level_rewards=torch.tensor(rewards, dtype=torch.float32).reshape(-1, 1),
                response_mask=torch.ones(16, 1), index=np.repeat(['a','b','c','d'],4),
                task_ids=['one']*16 if single else ['one']*8+['two']*8,
                dataset_ids=['data']*16, class_labels=['class']*16)

old=load_function(Path('/data/codefile/changshaoqing/human_behavior_atlas/verl/verl/trainer/ppo/core_algos.py'))
new=load_function(ROOT/'patched/core_algos.py')
sequence=[([0,0,0,1]*2+[0,1,2,3]*2), ([0,1,0,1]*2+[0,0,0,1]*2), ([0]*8+[0,1,2,3]*2)]
reset(); expected=[old(**args(v))[0].clone() for v in sequence]
reset(); actual=[new(**args(v))[0].clone() for v in sequence]
assert all(torch.equal(a,b) for a,b in zip(expected,actual)), 'Default changed original HARPO'
print('PASS original vs patched default: bitwise equality across history')

reset(); base90=[]
for v in sequence:
    base90.append(new(**args(v),xrpo_beta=.9)[0].clone())
reset()
for step,v in enumerate(sequence):
    metrics={}; out=new(**args(v),xrpo_beta=.9,xrpo_reverse_reference=True,xrpo_metrics=metrics)[0]
    post=h.get_latest_advantage_data()['post_grpo']['q2_advantages']
    refs=[math.sqrt(max(np.mean(np.abs(post[a])),1e-6)*max(np.mean(np.abs(post[b])),1e-6))
          for a,b in [('a','b'),('c','d')]]
    global_ref=math.sqrt(refs[0]*refs[1]); xs=[r/global_ref for r in refs]
    for task,x in zip(['one','two'],xs):
        assert math.isclose(metrics[f'xrpo/task/{task}/x'],x,rel_tol=1e-10)
    expected=base90[step]*torch.tensor([xs[0]]*8+[xs[1]]*8).view(-1,1)
    torch.testing.assert_close(out,expected,atol=1e-6,rtol=1e-6)
    assert torch.isfinite(out).all()
print('PASS x direction, exact geometric population, no added EMA/clamp, zero-group finite')

reset(); metrics={}
new(**args(sequence[0],single=True),xrpo_beta=.9,xrpo_reverse_reference=True,xrpo_metrics=metrics)
assert metrics['xrpo/task/one/x']==1.0
reset(); z=new(**args([0]*16),xrpo_beta=.9,xrpo_reverse_reference=True)[0]
assert torch.equal(z,torch.zeros_like(z))
print('PASS single-task x=1 and all-zero advantages stay zero')

reset()
new(**args(sequence[0]),xrpo_beta=.9)
prev=h.task_stats['one']['mixture_rho_t_ema']
prev_log=h.task_stats['one']['mixture_log_mult_ema_scale']
new(**args(sequence[1]),xrpo_beta=.9)
stats=h.task_stats['one']
assert math.isclose(stats['mixture_rho_t_ema'],.9*prev+.1*stats['mixture_rho_t_batch'])
assert math.isclose(stats['mixture_log_mult_ema_scale'],.9*prev_log+.1*stats['mixture_log_mult_inst_scale'])
print('PASS both actual EMA updates use new-observation share 10%')
