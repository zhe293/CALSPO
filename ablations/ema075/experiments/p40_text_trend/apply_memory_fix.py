"""Allow layer/LoRA FSDP wrapping for the P40 HF-only experiment."""
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
old='        if self._is_rollout and self.config.rollout.name == "hf":\n'
new='        if self._is_rollout and self.config.rollout.name == "hf" and not os.environ.get("HBA_EXPERIMENT_SEED"):\n'
if new not in source:
    assert source.count(old)==1
    source=source.replace(old,new)
    anchor='                actor_module = get_peft_model(actor_module, LoraConfig(**lora_config))\n'
    assert source.count(anchor)==1
    source=source.replace(anchor,anchor+'''                if os.environ.get("HBA_EXPERIMENT_SEED"):
                    for parameter in actor_module.parameters():
                        if parameter.requires_grad:
                            parameter.data = parameter.data.float()
                    print("HBA precision: frozen base", torch_dtype, "trainable LoRA float32")
''')
    ast.parse(source)
    backup=args.repo/'experiments/p40_text_trend/patch_backups'
    backup.mkdir(exist_ok=True)
    saved=backup/(path.name+'.'+hashlib.sha256(original).hexdigest()+'.bak')
    if not saved.exists():
        saved.write_bytes(original)
    path.write_text(source,encoding='utf-8')
print('memory_fix=PASS')
