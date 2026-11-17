"""Real installed Omni forward/backward using tiny random weights and real media.

This checks API, positions, audio/video token alignment, and differentiability.
It does not establish full 3B FSDP GPU feasibility.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
from omegaconf import OmegaConf
from transformers import AutoConfig, AutoProcessor
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniThinkerForConditionalGeneration as Thinker
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn

torch.set_num_threads(1)
path='/share/155/pretrained/Qwen2.5-Omni-3B'
p=AutoProcessor.from_pretrained(path,local_files_only=True)
c=AutoConfig.from_pretrained(path,local_files_only=True).thinker_config
for key,val in dict(hidden_size=32,intermediate_size=64,num_attention_heads=2,num_key_value_heads=2,num_hidden_layers=1,rope_scaling={'rope_type':'default','mrope_section':[2,3,3]}).items():
    setattr(c.text_config,key,val)
for key,val in dict(d_model=32,encoder_attention_heads=2,encoder_ffn_dim=64,encoder_layers=1,num_hidden_layers=1,output_dim=32).items():
    setattr(c.audio_config,key,val)
for key,val in dict(depth=1,hidden_size=32,embed_dim=32,intermediate_size=64,num_heads=2,out_hidden_size=32,fullatt_block_indexes=[0]).items():
    setattr(c.vision_config,key,val)
c._attn_implementation='eager'
model=Thinker(c).float().eval()
config=OmegaConf.create(dict(prompt_key='problem',image_key='images',video_key='videos',audio_key='audios',modalities='audio,videos',max_prompt_length=896,filter_overlong_prompts=False,truncation='error',return_multi_modal_inputs=True,format_prompt='/data/codefile/changshaoqing/human_behavior_atlas_xrpo/training/rl/format_prompt/default.jinja'))
ds=RLHFDataset(['/data/datasets/human_behavior_atlas-data/changshaoqing_multimodal_v4/train.parquet'],p.tokenizer,config,processor=p)
for sig in ('text','text_audio','text_video_audio'):
    row=ds[next(i for i,r in enumerate(ds.dataframe) if r['modality_signature']==sig)]
    mask=row['attention_mask'].bool()
    ids=row['input_ids'][mask].unsqueeze(0)
    am=torch.ones_like(ids)
    pos=row['position_ids'][:,mask].unsqueeze(1)
    media=row['multi_modal_inputs']
    with torch.no_grad():
        supplied=model(input_ids=ids,attention_mask=am,position_ids=pos,**media,use_cache=False).logits[:,-1].clone()
        inferred=model(input_ids=ids,attention_mask=am,**media,use_cache=False).logits[:,-1].clone()
        torch.testing.assert_close(supplied,inferred,atol=1e-6,rtol=1e-5)
    out=model(input_ids=ids,attention_mask=am,position_ids=pos,**media,use_cache=False)
    loss=out.logits[:,-1].float().square().mean()
    loss.backward()
    assert torch.isfinite(loss) and model.lm_head.weight.grad is not None
    assert torch.isfinite(model.lm_head.weight.grad).all()
    model.zero_grad(set_to_none=True)
    del out
    print('REAL_OMNI_FORWARD_BACKWARD_PASS',sig,'tokens',ids.shape[1],flush=True)
