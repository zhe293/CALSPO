"""Short real-model regression for repeated FSDP/LoRA backward passes.

Synthetic responses avoid the 29-minute rollout. Diagnostic only, never a result.
"""
import os
import json
from pathlib import Path
import socket
import sys
os.environ['VERL_ATTENTION_BACKEND']='eager'
os.environ['TOKENIZERS_PARALLELISM']='false'
print('DIAGNOSTIC starting imports',flush=True)

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
sys.path[:0]=[str(HERE),str(REPO/'verl')]
from resource_monitor import assert_free_gpus
selected_gpu=os.environ['CUDA_VISIBLE_DEVICES']
assert selected_gpu.isdigit(), 'Diagnostic requires exactly one explicitly selected GPU'
assert_free_gpus({int(selected_gpu)})
root=Path(os.environ['HBA_RUN_DIR'])
root.mkdir(parents=True,exist_ok=True)
with socket.socket() as sock:
    sock.bind(('127.0.0.1',0))
    port=sock.getsockname()[1]
os.environ.update(RANK='0',WORLD_SIZE='1',LOCAL_RANK='0',LOCAL_WORLD_SIZE='1',
                  MASTER_ADDR='127.0.0.1',MASTER_PORT=str(port),DISABLE_WORKER_INIT='1')
import torch
from hydra import compose,initialize_config_dir
from omegaconf import OmegaConf
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl import DataProto
from text_dataset import TextTrendDataset
from verl.utils.dataset.rl_dataset import collate_fn

manifest=json.loads(Path('/data/codefile/changshaoqing/harpo_p40_runs/experiments_gpu0_v4/grpo_reward_a/run_manifest.json').read_text())
overrides=manifest['command'][3:]
overrides=[x.replace('use_orig_params=true','use_orig_params=false') for x in overrides]
with initialize_config_dir(config_dir=str(REPO/'verl/verl/trainer/config'),version_base=None):
    config=compose(config_name='ppo_trainer',overrides=overrides)
    OmegaConf.resolve(config)
config.actor_rollout_ref.actor.optim.total_training_steps=16
worker=ActorRolloutRefWorker(config.actor_rollout_ref,role='actor_rollout')
worker.init_model()
dataset=TextTrendDataset(manifest['train'],worker.tokenizer,config.data,worker.processor)
rows=collate_fn([dataset[0],dataset[1]])
prompt=rows['input_ids'].cuda()
mask=rows['attention_mask'].cuda()
positions=rows['position_ids'].cuda()
# Check actual HF generation setup with only eight tokens, then exercise full
# production sequence length using deterministic diagnostic responses.
worker.rollout.config.response_length=8
mini=DataProto.from_dict(tensors={'input_ids':prompt[:1],'attention_mask':mask[:1],'position_ids':positions[:1]})
mini.meta_info.update(eos_token_id=worker.tokenizer.eos_token_id,pad_token_id=worker.tokenizer.pad_token_id)
generated=worker.rollout.generate_sequences(mini)
assert generated.batch['responses'].shape==(1,8)
print('DIAGNOSTIC generation=PASS',flush=True)
responses=torch.full((2,256),worker.tokenizer.encode('hello',add_special_tokens=False)[0],device='cuda',dtype=torch.long)
response_mask=torch.ones_like(responses)
tensor_data={'input_ids':torch.cat([prompt,responses],-1),'responses':responses,
             'attention_mask':torch.cat([mask,response_mask],-1),
             'position_ids':torch.cat([positions,positions[:,-1:]+torch.arange(1,257,device='cuda')],-1),
             'response_mask':response_mask,'advantages':torch.ones_like(responses,dtype=torch.float32)}
batch=DataProto.from_dict(tensors=tensor_data)
batch.meta_info.update(micro_batch_size=1,temperature=1.0,use_dynamic_bsz=False)
worker.actor.config.ppo_mini_batch_size=2
trainable=[p for p in worker.actor_module_fsdp.parameters() if p.requires_grad]
before=[p.detach().cpu().clone() for p in trainable]
for iteration in range(2):
    old,_=worker.actor.compute_log_prob(batch)
    batch.batch['old_log_probs']=old.detach()
    metrics=worker.actor.update_policy(batch)
    print('DIAGNOSTIC update='+str(iteration+1)+' metrics='+str(metrics),flush=True)
assert any(not torch.equal(a,p.detach().cpu()) for a,p in zip(before,trainable)), 'LoRA did not change'
assert all(torch.isfinite(p).all() for p in trainable)
generated=worker.rollout.generate_sequences(mini)
assert generated.batch['responses'].shape==(1,8)
result={'passed':True,'updates':2,'backward_microbatches':4,'sequence_length':768,
        'use_orig_params':False,'synthetic_diagnostic_not_experiment':True}
(root/'result.json').write_text(json.dumps(result,indent=2))
print('FSDP_GPU_REGRESSION=PASS',flush=True)
torch.distributed.destroy_process_group()
