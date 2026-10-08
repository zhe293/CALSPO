"""CPU-only verification of data, media and preserved source fingerprints."""
import hashlib,json,ast,os,argparse,subprocess
from pathlib import Path
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
parser=argparse.ArgumentParser()
parser.add_argument('--root',type=Path,default=ROOT)
ROOT=parser.parse_args().root.resolve()
def verify_media(parquet):
    # Same base_dir and path join as HBASFTBamDataset (lines 134, 446/459/480).
    base_dir=os.path.dirname(os.path.abspath(parquet));count=0
    rows=pq.read_table(parquet).to_pylist()
    for index,row in enumerate(rows):
        for column in ['images','videos','audios','ext_video_feats','ext_audio_feats']:
            for ref in row.get(column) or []:
                if not isinstance(ref,str) or os.path.isabs(ref) or '\\' in ref:
                    raise ValueError(f'{parquet.name} row {index} {column}: nonportable path {ref}')
                path=Path(os.path.join(base_dir,ref)).resolve()
                if not path.is_relative_to(Path(base_dir).resolve()):
                    raise ValueError(f'{parquet.name} row {index}: path escapes dataset directory')
                if not path.is_file() or path.stat().st_size==0:
                    raise FileNotFoundError(f'{parquet.name} row {index} {column}: {ref} -> {path}')
                count+=1
    return {'rows':len(rows),'media_references':count}
resolved={name:verify_media(ROOT/'datasets'/f'{name}.parquet') for name in ['train','validation','eval328']}
manifest=json.loads((ROOT/'datasets/asset_manifest.json').read_text(encoding='utf8'))
for name,item in manifest.items():
    p=ROOT/name.replace('\\','/');assert p.is_file(),name
    assert sha(p)==item['sha256'],name
    if 'rows' in item:assert pq.read_table(p).num_rows==item['rows'],name
sources=json.loads((ROOT/'datasets/source_manifest.json').read_text(encoding='utf8'))
for key,item in sources.items():
    assert sha(ROOT/'datasets/original'/f'{key}.parquet')==item['sha256'],key
tracked=None
if (ROOT/'.git').exists():
    tracked=set(subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode('utf8').split('\0'))
for variant in ['calspo','ema075','ema095']:
    code=ROOT if variant=='calspo' else ROOT/'ablations'/variant
    m=json.loads((code/'provenance/source_run_manifest.json').read_text(encoding='utf8'))
    for original,expected in m['code_sha256'].items():
        marker='/verl/verl/' if '/verl/verl/' in original else '/experiments/'
        prefix=original.split(marker)[0]
        p=code/Path(original).relative_to(prefix)
        if tracked is not None:
            assert p.relative_to(ROOT).as_posix() in tracked,('Not in version control',str(p))
        assert sha(p)==expected,(variant,str(p))
        ast.parse(p.read_text(encoding='utf8'))
    ray=(code/'verl/verl/trainer/ppo/ray_trainer.py').read_text(encoding='utf8')
    if variant!='ema095':assert 'xrpo_beta=0.75' in ray
    else:
        core=(code/'verl/verl/trainer/ppo/core_algos.py').read_text(encoding='utf8')
        assert 'xrpo_beta=0.75' not in ray and 'xrpo_beta: float = 0.95' in core
print(json.dumps({'status':'PASS','loader_resolved_datasets':resolved,
                  'media_files':sum(k.replace('\\','/').startswith('datasets/media/') for k in manifest),
                  'code_snapshots':3},indent=2))
