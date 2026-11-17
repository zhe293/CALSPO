"""Execute upstream reward code with only CPU model cache and metadata adaptation."""
import importlib.util
import os
from functools import lru_cache
from pathlib import Path

@lru_cache(maxsize=1)
def embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(os.environ['HBA_REWARD_EMBEDDING_MODEL'],device='cpu',local_files_only=True)

@lru_cache(maxsize=2)
def original(which):
    filename='human_behaviour.py' if which=='a' else 'human_behaviour_harpo.py'
    path=Path(__file__).resolve().parents[2]/'training/rl/reward_function'/filename
    spec=importlib.util.spec_from_file_location('hba_original_'+which,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.SentenceTransformer=lambda *a,**kw: embedding_model()
    return module

def score(which,data_sources,solution_strs,ground_truths,extra_infos,**kwargs):
    if not (len(data_sources)==len(solution_strs)==len(ground_truths)==len(extra_infos)):
        raise ValueError('Reward input length mismatch')
    results=original(which).human_behaviour_compute_score_batch(data_sources,solution_strs,ground_truths,extra_infos,**kwargs)
    if len(results)!=len(solution_strs):
        raise ValueError('Reward output length mismatch')
    for result in results:
        result.pop('task_type',None)
    return results
