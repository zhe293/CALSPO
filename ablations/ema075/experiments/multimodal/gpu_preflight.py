"""Actual 3B FSDP/LoRA multimodal gate. Separate disposable diagnostic weights."""
import os, json, socket
from pathlib import Path
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from verl import DataProto
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from resource_monitor import assert_free_gpus

repo=Path(__file__).resolve().parents[2]
root=Path(os.environ['HBA_RUN_DIR'])
plan=json.loads((root/'plan.json').read_text())
assert_free_gpus({int(os.environ['CUDA_VISIBLE_DEVICES'])})
with socket.socket() as sock:
    sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]
os.environ.update(RANK='0', WORLD_SIZE='1', LOCAL_RANK='0', LOCAL_WORLD_SIZE='1',
                  MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port), DISABLE_WORKER_INIT='1')
with initialize_config_dir(config_dir=str(repo/'verl/verl/trainer/config'),version_base=None):
    cfg=compose(config_name='ppo_trainer',overrides=plan['command'][3:])
    OmegaConf.resolve(cfg)
cfg.actor_rollout_ref.actor.optim.total_training_steps=38
worker=ActorRolloutRefWorker(cfg.actor_rollout_ref,role='actor_rollout')
worker.init_model()
ds=RLHFDataset(cfg.data.train_files,worker.tokenizer,cfg.data,worker.processor)
worker.rollout.config.response_length=8
worker.actor.config.ppo_mini_batch_size=1
trainable=[p for p in worker.actor_module_fsdp.parameters() if p.requires_grad]
before=[p.detach().cpu().clone() for p in trainable]
for sig in ('text','text_audio','text_video_audio'):
    # CPU audit records the largest token example for each modality.
    choices=json.loads((Path(__file__).parent/'cpu_audit_result.json').read_text())['max_indices']
    row=ds[choices[sig]]
    batch=DataProto.from_single_dict(collate_fn([row])).to('cuda')
    mini=batch.select(batch_keys=['input_ids','attention_mask','position_ids'],
                      non_tensor_batch_keys=['multi_modal_inputs','modality_signature'])
    mini.meta_info.update(eos_token_id=worker.tokenizer.eos_token_id,pad_token_id=worker.tokenizer.pad_token_id)
    generated=worker.rollout.generate_sequences(mini)
    assert generated.batch['responses'].shape==(1,8)
    prompt=batch.batch['input_ids']; mask=batch.batch['attention_mask']; pos=batch.batch['position_ids']
    responses=torch.full((1,256),worker.tokenizer.encode('hello',add_special_tokens=False)[0],device='cuda',dtype=torch.long)
    rm=torch.ones_like(responses)
    response_pos=pos.amax(dim=(1,2),keepdim=True)+torch.arange(1,257,device='cuda').view(1,1,-1)
    tensors=dict(input_ids=torch.cat([prompt,responses],-1), responses=responses,
                 attention_mask=torch.cat([mask,rm],-1),
                 position_ids=torch.cat([pos,response_pos.expand(-1,3,-1)],-1),
                 response_mask=rm, advantages=torch.ones_like(responses,dtype=torch.float32))
    train=DataProto.from_dict(tensors=tensors, non_tensors={'multi_modal_inputs':batch.non_tensor_batch['multi_modal_inputs']})
    train.meta_info.update(micro_batch_size=1,temperature=1.0,use_dynamic_bsz=False)
    old,_=worker.actor.compute_log_prob(train)
    assert torch.isfinite(old).all()
    train.batch['old_log_probs']=old.detach()
    worker.actor.update_policy(train)
    assert all(torch.isfinite(p).all() for p in trainable)
    print('GPU_MULTIMODAL_PASS',sig,flush=True)
assert any(not torch.equal(a,p.detach().cpu()) for a,p in zip(before,trainable))
(root/'result.json').write_text(json.dumps({'passed':True,'diagnostic_only':True,'modalities':3,'response_length':256}))
torch.distributed.destroy_process_group()
