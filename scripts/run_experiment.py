"""Portable launcher for the three preserved implementation snapshots.

Prints the resolved command by default; --execute explicitly starts a GPU job.
"""
import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--variant',choices=['calspo','ema075','ema095'],default='calspo')
    p.add_argument('--mode',choices=['train','evaluate'],default='train')
    p.add_argument('--base-model',required=True);p.add_argument('--embedding-model',required=True)
    p.add_argument('--gpu',required=True,help='Explicit physical GPU index')
    p.add_argument('--seed',type=int,default=48);p.add_argument('--output',required=True)
    p.add_argument('--checkpoint',help='Complete global_step_N checkpoint directory for evaluation')
    p.add_argument('--execute',action='store_true');a=p.parse_args()
    code=ROOT if a.variant=='calspo' else ROOT/'ablations'/a.variant
    m=json.loads((code/'provenance/source_run_manifest.json').read_text(encoding='utf8'))
    base=Path(a.base_model).resolve();embedding=Path(a.embedding_model).resolve();output=Path(a.output).resolve()
    if a.mode=='evaluate' and not a.checkpoint:p.error('--checkpoint is required for evaluation')
    old_repo=next(k.split('/verl/verl/')[0] for k in m['code_sha256'] if '/verl/verl/' in k)
    overrides={
        'data.train_files':str(ROOT/'datasets/train.parquet'),
        'data.val_files':str(ROOT/'datasets'/('eval328.parquet' if a.mode=='evaluate' else 'validation.parquet')),
        '+data.seed':str(a.seed),'actor_rollout_ref.model.path':str(base),
        'trainer.default_local_dir':str(output),'trainer.validation_data_dir':str(output/'final_test'),
        'trainer.advantage_save_dir':str(output/'advantages'),
        'trainer.experiment_name':f'{a.mode}_{a.variant}',
        'trainer.resume_mode':'resume_path' if a.mode=='evaluate' else 'disable',
        'trainer.val_before_train':'true' if a.mode=='evaluate' else 'false',
    }
    command=[sys.executable]+m['command'][1:]
    seen=set()
    for i,arg in enumerate(command):
        if '=' not in arg:continue
        key,value=arg.split('=',1);seen.add(key)
        if key in overrides:value=overrides[key]
        elif value.startswith(old_repo+'/'):value=str(code/Path(value).relative_to(old_repo))
        command[i]=key+'='+value
    if a.mode=='evaluate':
        command+=['trainer.resume_from_path='+str(Path(a.checkpoint).resolve()),'trainer.val_only=true','trainer.del_local_ckpt_after_load=false']
    env=os.environ.copy()
    for key in list(env):
        if key.startswith(('HBA_XRPO','HBA_LABEL','HBA_AVSPO')):env.pop(key)
    env.update(CUDA_VISIBLE_DEVICES=a.gpu,HBA_AVSPO_HARPO='1',HBA_REWARD_EMBEDDING_MODEL=str(embedding),
               HBA_EXPERIMENT_SEED=str(a.seed),HBA_RUN_DIR=str(output),PYTHONHASHSEED=str(a.seed),
               PYTHONUNBUFFERED='1',WANDB_MODE='disabled',RAY_ADDRESS='local',
               RAY_TMPDIR=str(output/'ray_tmp'),VERL_ATTENTION_BACKEND='eager',
               HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',NUMEXPR_NUM_THREADS='1',
               OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
               HBA_LABEL_SFT='1' if a.variant=='calspo' else '0',HBA_LABEL_SFT_LAMBDA='1.0',
               HBA_LABEL_CLASS_WEIGHTS=json.dumps(m.get('label_supervision',{}).get('class_weights',{})),
               TOKENIZERS_PARALLELISM='false')
    env['PYTHONPATH']=os.pathsep.join(str(code/x) for x in ['experiments/xrpo_multimodal_304','experiments/p40_text_trend','verl'])
    print(json.dumps({'variant':a.variant,'seed':a.seed,'working_directory':str(ROOT),'command':command,
                      'environment':{k:v for k,v in env.items() if k.startswith('HBA_') or k in ['CUDA_VISIBLE_DEVICES','PYTHONPATH','VERL_ATTENTION_BACKEND','RAY_TMPDIR','PYTHONHASHSEED']}},ensure_ascii=False,indent=2))
    if a.execute:
        if os.name=='nt':p.error('Training requires Linux/CUDA; run this launcher on the configured server')
        for path in [base,embedding]:
            if not path.is_dir():p.error('Missing model directory: '+str(path))
        if a.mode=='train' and output.exists() and any(output.iterdir()):p.error('Training output must be new or empty')
        if a.mode=='evaluate' and not Path(a.checkpoint).is_dir():p.error('Checkpoint directory does not exist')
        # Check media against the dataset loader's base directory before any GPU job.
        subprocess.run([sys.executable,str(ROOT/'scripts/verify_assets.py')],cwd=ROOT,env=env,check=True)
        output.mkdir(parents=True,exist_ok=True)
        subprocess.run(command,cwd=ROOT,env=env,check=True)
if __name__=='__main__':main()
