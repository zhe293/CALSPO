"""Use tokenizer token IDs when an otherwise present generation config omits them."""
import argparse
import ast
import hashlib
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('--repo',type=Path,required=True)
args=parser.parse_args()
path=args.repo/'verl/verl/workers/fsdp_workers.py'
original=path.read_bytes()
source=original.decode('utf-8').replace('\r\n','\n')
for name in ('eos_token_id','pad_token_id'):
    old=f'''            "{name}": self.generation_config.{name}
            if self.generation_config is not None
            else self.tokenizer.{name},'''
    new=f'''            "{name}": self.generation_config.{name}
            if self.generation_config is not None and self.generation_config.{name} is not None
            else self.tokenizer.{name},'''
    if new not in source:
        assert source.count(old)==1
        source=source.replace(old,new)
ast.parse(source)
backup=args.repo/'experiments/p40_text_trend/patch_backups'
backup.mkdir(exist_ok=True)
saved=backup/(path.name+'.'+hashlib.sha256(original).hexdigest()+'.bak')
if not saved.exists():
    saved.write_bytes(original)
path.write_text(source,encoding='utf-8')
print('token_fix=PASS')
