"""CPU-only verification of data, media and preserved source fingerprints."""
import hashlib,json,ast
from pathlib import Path
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((ROOT/'datasets/asset_manifest.json').read_text(encoding='utf8'))
for name,item in manifest.items():
    p=ROOT/name;assert p.is_file(),name
    assert sha(p)==item['sha256'],name
    if 'rows' in item:assert pq.read_table(p).num_rows==item['rows'],name
for variant in ['calspo','ema075','ema095']:
    code=ROOT if variant=='calspo' else ROOT/'ablations'/variant
    m=json.loads((code/'provenance/source_run_manifest.json').read_text(encoding='utf8'))
    for original,expected in m['code_sha256'].items():
        marker='/verl/verl/' if '/verl/verl/' in original else '/experiments/'
        prefix=original.split(marker)[0]
        p=code/Path(original).relative_to(prefix)
        assert sha(p)==expected,(variant,str(p))
        ast.parse(p.read_text(encoding='utf8'))
    ray=(code/'verl/verl/trainer/ppo/ray_trainer.py').read_text(encoding='utf8')
    if variant!='ema095':assert 'xrpo_beta=0.75' in ray
    else:
        core=(code/'verl/verl/trainer/ppo/core_algos.py').read_text(encoding='utf8')
        assert 'xrpo_beta=0.75' not in ray and 'xrpo_beta: float = 0.95' in core
print('PASS: 304/216/328 rows, 471 media hashes, three code snapshots and EMA settings')
