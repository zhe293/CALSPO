"""Wrap ordinary samplers for the upstream trainer's batch_sampler interface."""
import argparse
import ast
import hashlib
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--repo', type=Path, required=True)
args = parser.parse_args()
path = args.repo / 'verl/verl/trainer/main_ppo.py'
original = path.read_bytes()
source = original.decode('utf-8').replace('\r\n', '\n')
old = '    return sampler\n'
new = '''    # HBA: trainer passes this object as batch_sampler, so yield index lists.
    if isinstance(sampler, (SequentialSampler, RandomSampler)):
        from torch.utils.data import BatchSampler
        batch_size = data_config.get("train_batch_size" if split == "train" else "val_batch_size")
        if batch_size is None:
            batch_size = len(dataset)
        sampler = BatchSampler(sampler, batch_size=int(batch_size), drop_last=(split == "train"))
    return sampler
'''
if new not in source:
    assert source.count(old) == 1
    source = source.replace(old, new)
    ast.parse(source)
    backup = args.repo / 'experiments/p40_text_trend/patch_backups'
    backup.mkdir(exist_ok=True)
    saved = backup / (path.name + '.' + hashlib.sha256(original).hexdigest() + '.bak')
    if not saved.exists():
        saved.write_bytes(original)
    path.write_text(source, encoding='utf-8')
print('batching_fix=PASS')
