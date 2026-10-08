"""Download the exact official base model and verify its weight shards."""
import argparse,json,hashlib
from pathlib import Path
from huggingface_hub import snapshot_download
root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--output',default=str(root/'models/Qwen2.5-Omni-3B'));p.add_argument('--embedding-output')
a=p.parse_args();m=json.loads((root/'models/base_model_manifest.json').read_text(encoding='utf8'))
snapshot_download(m['model_id'],revision=m['revision'],local_dir=a.output)
for item in m['files']:
    if item['name'].endswith('.safetensors'):
        path=Path(a.output)/item['name']
        h=hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
        if path.stat().st_size!=item['bytes'] or h.hexdigest()!=item['sha256']:raise RuntimeError('Model verification failed: '+item['name'])
print('Base model download and weight verification complete:',a.output)
if a.embedding_output:
    snapshot_download('sentence-transformers/all-MiniLM-L6-v2',local_dir=a.embedding_output)
    print('Reward embedding model downloaded:',a.embedding_output)
