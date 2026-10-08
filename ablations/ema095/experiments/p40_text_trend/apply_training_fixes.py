"""Checked, backed-up patches for the personal P40 checkout only."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--repo',type=Path,required=True)
    args=ap.parse_args()
    root=args.repo.resolve()
    changes={}
    def patch(rel,old,new,count=1):
        path=root/'verl/verl'/rel
        text=changes.get(path,path.read_text(encoding='utf-8'))
        if new in text:
            return
        if text.count(old)!=count:
            raise RuntimeError(f'{path}: expected {count} anchors, found {text.count(old)}')
        changes[path]=text.replace(old,new)

    patch('workers/fsdp_workers.py','rank_seed = experiment_seed + self.rank','rank_seed = experiment_seed  # identical initial LoRA on every rank')
    patch('workers/fsdp_workers.py','param_dtype = torch.bfloat16',
          'param_dtype = torch.float16 if os.environ.get("HBA_EXPERIMENT_SEED") else torch.bfloat16',count=2)
    patch('workers/rollout/hf_rollout.py','"num_return_sequences": self.config.n',
          '"num_return_sequences": 1,  # trainer already repeats each prompt n times\n                # HBA_P40_SINGLE_REPEAT')
    patch('workers/rollout/hf_rollout.py','        self.module = module\n',
          '        self.module = module\n'
          '        import os\n'
          '        if os.environ.get("HBA_EXPERIMENT_SEED"):\n'
          '            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0\n'
          '            torch.manual_seed(int(os.environ["HBA_EXPERIMENT_SEED"]) + rank)\n')
    patch('workers/actor/dp_actor.py','        self.device_name = get_device_name()\n',
          '        self.device_name = get_device_name()\n'
          '        from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler\n'
          '        self._p40_fp16 = self.device_name == "cuda" and not torch.cuda.is_bf16_supported()\n'
          '        self._p40_scaler = ShardedGradScaler(enabled=self._p40_fp16, init_scale=256.0)\n')
    patch('workers/actor/dp_actor.py','with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):',
          'with torch.autocast(device_type=self.device_name, dtype=torch.float16 if self._p40_fp16 else torch.bfloat16):')
    patch('workers/actor/dp_actor.py','                    loss.backward()','                    self._p40_scaler.scale(loss).backward()')
    patch('workers/actor/dp_actor.py','        assert self.config.grad_clip is not None\n',
          '        assert self.config.grad_clip is not None\n        self._p40_scaler.unscale_(self.actor_optimizer)\n')
    patch('workers/actor/dp_actor.py','        # if grad_norm is not finite, skip the update\n'
          '        if not torch.isfinite(grad_norm):\n'
          '            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")\n'
          '            self.actor_optimizer.zero_grad()\n'
          '        else:\n'
          '            self.actor_optimizer.step()\n'
          '        return grad_norm',
          '        if not torch.isfinite(grad_norm):\n'
          '            raise FloatingPointError("Nonfinite gradient: abort controlled run instead of silently counting skipped updates")\n'
          '        self._p40_scaler.step(self.actor_optimizer)\n'
          '        self._p40_scaler.update()\n'
          '        return grad_norm')

    patch('trainer/ppo/ray_trainer.py','import wandb\n',
          'import wandb\nfrom recording import phase, record_step\nfrom text_evaluation import evaluate as evaluate_text_trend\n')
    patch('trainer/ppo/ray_trainer.py','    def _validate(self):\n',
          '    def _validate(self):\n        phase("evaluation", self.global_steps)\n')
    patch('trainer/ppo/ray_trainer.py','    def _save_checkpoint(self):\n',
          '    def _save_checkpoint(self):\n        phase("checkpoint", self.global_steps)\n')
    old='        metrics = compute_metrics_by_data_source(all_predictions, all_ground_truths, \n                                                 all_datasets, val_data_dir, self.global_steps)\n\n        wandb.log(metrics, step=self.global_steps)'
    new='        metrics = evaluate_text_trend(all_predictions, all_ground_truths,\n                                      all_data_sources, val_data_dir, self.global_steps)\n\n        if "wandb" in self.config.trainer.logger and wandb.run is not None:\n            wandb.log(metrics, step=self.global_steps)'
    patch('trainer/ppo/ray_trainer.py',old,new)
    patch('trainer/ppo/ray_trainer.py','        return metric_dict\n','        metric_dict.update(metrics)\n        return metric_dict\n')
    # Final validation is performed after saving and explicitly reloading the final weights.
    old='''                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with marked_timer("testing", timing_raw, color="green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)
'''
    patch('trainer/ppo/ray_trainer.py',old,'                    # HBA final validation moved after checkpoint reload.\n')
    patch('trainer/ppo/ray_trainer.py','                            self._save_checkpoint()\n',
          '                            self._save_checkpoint()\n'
          '                    if is_last_step:\n'
          '                        phase("reload", self.global_steps)\n'
          '                        actor_path = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}", "actor")\n'
          '                        self.actor_rollout_wg.load_checkpoint(actor_path, del_local_after_load=False)\n'
          '                        with open(os.path.join(self.config.trainer.default_local_dir, "checkpoint_reloaded.json"), "w") as f:\n'
          '                            import json\n'
          '                            json.dump({"step": self.global_steps, "path": actor_path}, f)\n'
          '                        with marked_timer("testing", timing_raw, color="green"):\n'
          '                            last_val_metrics = self._validate()\n'
          '                        metrics.update(last_val_metrics)\n')
    patch('trainer/ppo/ray_trainer.py','                with marked_timer("step", timing_raw):\n',
          '                phase("train", self.global_steps)\n                with marked_timer("step", timing_raw):\n')
    patch('trainer/ppo/ray_trainer.py','                logger.log(data=metrics, step=self.global_steps)\n',
          '                record_step(metrics, batch, self.global_steps)\n                phase("between_steps", self.global_steps)\n                logger.log(data=metrics, step=self.global_steps)\n')

    # Parse all candidate files before making any change.
    for path,text in changes.items():
        ast.parse(text,filename=str(path))
    backup=root/'experiments/p40_text_trend/patch_backups'
    backup.mkdir(exist_ok=True)
    manifest={}
    for path,text in changes.items():
        original=path.read_bytes()
        digest=hashlib.sha256(original).hexdigest()
        saved=backup/(path.name+'.'+digest+'.bak')
        if not saved.exists():
            saved.write_bytes(original)
        path.write_text(text,encoding='utf-8')
        manifest[str(path.relative_to(root))]={'before':digest,'after':hashlib.sha256(path.read_bytes()).hexdigest()}
    (backup/'last_apply.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('training_fixes=PASS files='+str(len(changes)))

if __name__=='__main__':
    main()
