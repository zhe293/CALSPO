"""Use the installed Omni model's actual position routine without model weights."""
from functools import lru_cache
from types import SimpleNamespace, MethodType
from transformers import AutoConfig
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniThinkerForConditionalGeneration as Thinker

@lru_cache(maxsize=2)
def position_context(model_path):
    cfg = AutoConfig.from_pretrained(model_path, local_files_only=True).thinker_config
    obj = SimpleNamespace(config=cfg, spatial_merge_size=cfg.vision_config.spatial_merge_size)
    obj.get_llm_pos_ids_for_vision = MethodType(Thinker.get_llm_pos_ids_for_vision, obj)
    obj.get_chunked_index = MethodType(Thinker.get_chunked_index, obj)
    return obj

def omni_position_ids(model_path, input_ids, attention_mask, media):
    mask = media.get('feature_attention_mask')
    ids, _ = Thinker.get_rope_index(
        position_context(str(model_path)), input_ids=input_ids,
        attention_mask=attention_mask, image_grid_thw=media.get('image_grid_thw'),
        video_grid_thw=media.get('video_grid_thw'),
        audio_seqlens=mask.sum(dim=1) if mask is not None else None,
        second_per_grids=media.get('video_second_per_grid'),
        use_audio_in_video=False,
    )
    return ids.transpose(0, 1)  # model [3,batch,L] -> dataset [batch,3,L]
