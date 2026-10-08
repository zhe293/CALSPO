"""Checked P40 startup compatibility patch; preserve originals by hash."""
import argparse
import ast
import hashlib
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    args = parser.parse_args()
    path = args.repo / 'verl/verl/workers/actor/dp_actor.py'
    old = '    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input\n'
    new = '''    try:
        from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
    except ModuleNotFoundError as exc:
        if exc.name != "flash_attn":
            raise
        index_first_axis = pad_input = rearrange = unpad_input = None
'''
    original = path.read_bytes()
    source = original.decode('utf-8').replace('\r\n', '\n')
    if new in source:
        print('runtime_fixes=already_applied')
        return
    assert source.count(old) == 1
    source = source.replace(old, new)
    anchor = '        self.use_remove_padding = self.config.get("use_remove_padding", False)\n'
    assert source.count(anchor) == 1
    source = source.replace(anchor, anchor + '        if self.use_remove_padding and unpad_input is None:\n            raise ImportError("remove_padding requires flash_attn; P40 experiment must disable it")\n')
    ast.parse(source)
    backup = args.repo / 'experiments/p40_text_trend/patch_backups'
    backup.mkdir(exist_ok=True)
    saved = backup / (path.name + '.' + hashlib.sha256(original).hexdigest() + '.bak')
    if not saved.exists():
        saved.write_bytes(original)
    path.write_text(source, encoding='utf-8')
    print('runtime_fixes=PASS')


if __name__ == '__main__':
    main()
