"""Read-only CPU audit; no training, Ray initialization, or GPU model loading."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import sys
import json
import hashlib
import importlib.util
from pathlib import Path
from collections import Counter

repo = Path('/data/codefile/changshaoqing/human_behavior_atlas')
exp = repo / 'experiments/p40_text_trend'
sys.path.insert(0, str(repo / 'verl'))

def emit(name, value):
    print(json.dumps({'check': name, 'result': value}, ensure_ascii=False), flush=True)

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

import torch
import numpy as np
import pyarrow.parquet as pq
import transformers
import logging
logging.disable(logging.WARNING)
transformers.utils.logging.set_verbosity_error()
emit('versions', {'torch': torch.__version__, 'transformers': transformers.__version__})
emit('files', {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in exp.glob('*.py')})
sets = {}
rows_by_split = {}
for split in ('train', 'validation'):
    p = Path('/data/datasets/human_behavior_atlas-data/changshaoqing_text_trend') / (split + '.parquet')
    rows = pq.read_table(p).to_pylist()
    rows_by_split[split] = rows
    sets[split] = {str(r['problem']).strip() for r in rows}
    emit(split, {'rows': len(rows), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                 'tasks': dict(Counter(r['task'] for r in rows)),
                 'labels': dict(Counter(str((r['task'],r['answer'])) for r in rows)),
                 'duplicate_problem_count': len(rows) - len(sets[split]),
                 'media_rows': sum(bool(r.get('images') or r.get('videos') or r.get('audios')) for r in rows)})
emit('train_validation_problem_overlap', len(sets['train'] & sets['validation']))

model = Path('/share/155/pretrained/Qwen2.5-Omni-3B')
idx = json.loads((model / 'model.safetensors.index.json').read_text())
emit('missing_model_shards', [n for n in set(idx['weight_map'].values()) if not (model/n).is_file()])
from transformers import AutoProcessor
from jinja2 import Template
processor = AutoProcessor.from_pretrained(str(model), local_files_only=True)
from verl.utils.dataset.rl_dataset import RLHFDataset
d = RLHFDataset.__new__(RLHFDataset)
d.prompt_key='problem'
d.modalities=set()
d.image_key='images'
d.video_key='videos'
d.audio_key='audios'
d.format_prompt=Template((exp/'text_prompt.jinja').read_text())
d.processor=processor
d.tokenizer=processor.tokenizer
d.base_dir='/data/datasets/human_behavior_atlas-data/changshaoqing_text_trend'
d.max_prompt_length=512
d.truncation='error'
d.return_multi_modal_inputs=False
d.return_raw_chat=False
d.return_full_prompt=False
import copy
for split, rows in rows_by_split.items():
    d.dataframe=copy.deepcopy(rows)
    lengths=[]
    failures=[]
    for i,row in enumerate(rows):
        try:
            processed=d[i]
            lengths.append(len(processed['raw_prompt_ids']))
        except Exception as e:
            failures.append({'row':i,'error':type(e).__name__+': '+str(e)[:180]})
    emit(split+'_actual_prompt_path', {'success':len(lengths),'failures':len(failures),'examples':failures[:2],
                                      'max_tokens':max(lengths,default=0),'over_512':sum(n>512 for n in lengths)})
    lengths=[]
    for row in rows:
        content=d.format_prompt.render(content=row['problem'])
        messages=[{'role':'user','content':[{'type':'text','text':content}]}]
        rendered=processor.apply_chat_template(messages,add_generation_prompt=True,tokenize=False)
        if isinstance(rendered,list):
            rendered=rendered[0]
        lengths.append(len(processor.tokenizer.encode(rendered,add_special_tokens=False)))
    emit(split+'_intended_prompt_path',{'max_tokens':max(lengths),'over_512':sum(n>512 for n in lengths)})

b=load(exp/'reward_b_local.py','audit_reward_b')
try:
    value=b.human_behaviour_compute_score_batch(['x','x'],[r'\boxed{yes}',r'\boxed{no}'],['yes','no'],[None,None],
                                               task_ids=np.array(['anxiety_cls','depression_cls'],dtype=object))
    emit('reward_b_numpy_tasks',value)
except Exception as e:
    emit('reward_b_numpy_tasks',type(e).__name__+': '+str(e))

from verl.trainer.ppo import core_algos
from verl.trainer.ppo.ray_trainer import compute_advantage
from verl import DataProto
from tensordict import TensorDict
for algo in ('grpo','harpo'):
    try:
        batch=DataProto(batch=TensorDict({'token_level_rewards':torch.tensor([[1.,0.],[0.,0.],[1.,0.],[0.,0.]]),
                                        'response_mask':torch.ones(4,2)},batch_size=[4]),
                        non_tensor_batch={'uid':np.array(['a','a','b','b'],dtype=object),
                                          'task':np.array(['anxiety_cls']*2+['depression_cls']*2,dtype=object),
                                          'dataset':np.array(['audit']*4,dtype=object),
                                          'class_label':np.array(['yes']*4,dtype=object)})
        out=compute_advantage(batch,adv_estimator=algo,num_repeat=2)
        emit(algo+'_advantage',{'has_advantages':'advantages' in out.batch,
                               'finite':bool(torch.isfinite(out.batch['advantages']).all()) if 'advantages' in out.batch else False})
    except Exception as e:
        emit(algo+'_advantage',type(e).__name__+': '+str(e))

for rel, tokens in {
 'verl/verl/workers/fsdp_workers.py':['HBA_EXPERIMENT_SEED','param_dtype = torch.bfloat16','VERL_ATTENTION_BACKEND'],
 'verl/verl/workers/actor/dp_actor.py':['dtype=torch.bfloat16','use_torch_compile'],
 'verl/verl/workers/rollout/hf_rollout.py':['"num_return_sequences": self.config.n'],
 'verl/verl/trainer/ppo/ray_trainer.py':['wandb.log(metrics, step=self.global_steps)','gen_batch = gen_batch.repeat'],
}.items():
    content=(repo/rel).read_text()
    emit(rel,{t:t in content for t in tokens})
emit('audit_complete',True)
