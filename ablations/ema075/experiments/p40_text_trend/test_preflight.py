"""CPU regressions for the actual text, reward, rollout, configuration and monitor paths."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['OMP_NUM_THREADS']='2'
os.environ['HBA_REWARD_EMBEDDING_MODEL']='/share/155/pretrained/mashang/SAGEPO/reward_models/all-MiniLM-L6-v2'
import json
import logging
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
sys.path[:0]=[str(HERE),str(REPO/'verl')]
import numpy as np
import torch
import transformers
from omegaconf import OmegaConf
from transformers import AutoProcessor
from text_dataset import TextTrendDataset
logging.disable(logging.WARNING)
transformers.utils.logging.set_verbosity_error()
MODEL='/share/155/pretrained/Qwen2.5-Omni-3B'
DATA='/data/datasets/human_behavior_atlas-data/anonymous_text_trend'

class Preflight(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True)

    def test_dataset_all_rows_and_prompt_content(self):
        config=OmegaConf.create({'format_prompt':str(HERE/'text_prompt.jinja'),'max_prompt_length':512})
        counts={}
        for split,expected in [('train',128),('validation',32)]:
            ds=TextTrendDataset(DATA+'/'+split+'.parquet',self.processor.tokenizer,config,self.processor)
            self.assertEqual(len(ds),expected)
            truncated=0
            for i in range(len(ds)):
                row=ds[i]
                self.assertEqual(tuple(row['input_ids'].shape),(512,))
                self.assertEqual(tuple(row['position_ids'].shape),(512,))
                decoded=self.processor.tokenizer.decode(row['raw_prompt_ids'])
                self.assertIn('Respond in this exact form:',decoded)
                self.assertIn(str(ds.rows[i]['problem'])[:30],decoded)
                truncated+=row['extra_info']['truncated']
            counts[split]={'rows':len(ds),'truncated':truncated}
        print('dataset_checked='+json.dumps(counts),flush=True)

    def test_rewards_numpy_and_original_equivalence(self):
        from original_reward import score,original
        preds=[r'<think>x</think>\boxed{anxiety}',r'\boxed{ no anxiety }','x'*720,r'\boxed{depression}']
        gts=['anxiety','no anxiety','no anxiety','depression']
        tasks=np.array(['anxiety_cls']*3+['depression_cls'],dtype=object)
        for name in ('a','b'):
            args=(['d']*4,preds,gts,[None]*4)
            expected=original(name).human_behaviour_compute_score_batch(*args,task_ids=tasks)
            actual=score(name,*args,task_ids=tasks)
            for left,right in zip(actual,expected):
                for key,value in left.items():
                    self.assertAlmostEqual(float(value),float(right[key]),places=6)
                self.assertNotIn('task_type',left)

    def test_hf_rollout_exactly_one_output_per_repeated_input(self):
        from verl import DataProto
        import verl.workers.rollout.hf_rollout as hf
        hf.get_device_name=lambda:'cpu'
        from types import SimpleNamespace
        hf.get_torch_device=lambda:SimpleNamespace(empty_cache=lambda:None)
        class Fake(torch.nn.Module):
            def generate(self,input_ids,**kw):
                self.n=kw['generation_config'].num_return_sequences
                from types import SimpleNamespace
                values=input_ids.repeat_interleave(self.n,dim=0)
                return SimpleNamespace(sequences=torch.cat([values,torch.full((len(values),2),3)],dim=1))
        model=Fake()
        config=OmegaConf.create({'do_sample':True,'temperature':1.0,'response_length':2,'top_p':1.0,
                                 'top_k':0,'n':4,'micro_batch_size':1})
        rollout=hf.HFRollout(model,config)
        prompts=DataProto.from_dict(tensors={'input_ids':torch.ones(8,3,dtype=torch.long),
                                             'attention_mask':torch.ones(8,3,dtype=torch.long),
                                             'position_ids':torch.arange(3).repeat(8,1)},
                                    meta_info={'eos_token_id':2,'pad_token_id':0})
        output=rollout.generate_sequences(prompts)
        self.assertEqual(len(output),8)
        self.assertEqual(model.n,1)

    def test_evaluation_counts_invalid_and_minority(self):
        from text_evaluation import evaluate
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            evaluate(['no anxiety']*3+['nonsense'],['no anxiety']*3+['anxiety'],['anxiety_cls']*4,tmp,1)
            result=json.loads((Path(tmp)/'metrics_step_1.json').read_text())['per_task']['anxiety_cls']
            self.assertEqual(result['accuracy'],.75)
            self.assertEqual(result['positive_recall'],0)
            self.assertEqual(result['balanced_accuracy'],.5)
            self.assertEqual(result['invalid_rate'],.25)

    def test_grpo_harpo_real_dispatch(self):
        from verl import DataProto
        from verl.trainer.ppo.ray_trainer import compute_advantage
        for algorithm in ('grpo','harpo'):
            batch=DataProto.from_dict(tensors={'token_level_rewards':torch.tensor([[1.,0.],[0.,0.],[1.,0.],[0.,0.]]),
                                               'response_mask':torch.ones(4,2)},
                                     non_tensors={'uid':np.array(['a','a','b','b'],dtype=object),
                                                  'task':np.array(['anxiety_cls']*2+['depression_cls']*2,dtype=object),
                                                  'dataset':np.array(['audit']*4,dtype=object),
                                                  'class_label':np.array(['yes']*4,dtype=object)})
            result=compute_advantage(batch,adv_estimator=algorithm,num_repeat=2)
            self.assertTrue(torch.isfinite(result.batch['advantages']).all())

    def test_missing_checkpoint_cannot_pass_completion(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            result=subprocess.run([sys.executable,str(HERE/'verify_run_complete.py'),'--run-dir',tmp,
                                   '--expected-steps','1'],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertNotIn('run_complete=PASS',result.stdout)

    def test_three_hydra_configs_and_seed(self):
        import shlex
        from hydra import compose,initialize_config_dir
        configs=[]
        for run in ('grpo_reward_a','harpo_reward_a','harpo_reward_b'):
            cmd=[sys.executable,str(HERE/'run_experiment.py'),'--run',run,'--repo',str(REPO),'--model',MODEL,
                 '--reward-model',os.environ['HBA_REWARD_EMBEDDING_MODEL'],'--train',DATA+'/train.parquet',
                 '--validation',DATA+'/validation.parquet','--output-root',str(HERE/'never_execute'),
                 '--gpus','2','--total-steps','16']
            result=subprocess.check_output(cmd,text=True)
            line=next(line for line in result.splitlines() if 'verl.trainer.main_ppo' in line)
            overrides=shlex.split(line)[3:]
            with initialize_config_dir(config_dir=str(REPO/'verl/verl/trainer/config'),version_base=None):
                config=compose(config_name='ppo_trainer',overrides=overrides)
                OmegaConf.resolve(config)
            self.assertEqual(config.data.seed,45)
            self.assertEqual(config.trainer.total_training_steps,16)
            self.assertEqual(config.ray_init.num_cpus,4)
            self.assertFalse(config.actor_rollout_ref.actor.fsdp_config.use_orig_params)
            self.assertGreaterEqual(config.trainer.total_epochs,2)
            self.assertFalse(config.actor_rollout_ref.actor.use_torch_compile)
            self.assertFalse(config.actor_rollout_ref.model.use_remove_padding)
            configs.append(config)
        self.assertEqual(configs[0].custom_reward_function.path,configs[1].custom_reward_function.path)
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer
        from verl.utils.dataset.rl_dataset import collate_fn
        for config in configs:
            trainer = RayPPOTrainer.__new__(RayPPOTrainer)
            trainer.config = config
            trainer.current_epoch = 0
            train = TextTrendDataset(DATA+'/train.parquet',self.processor.tokenizer,config.data,self.processor)
            val = TextTrendDataset(DATA+'/validation.parquet',self.processor.tokenizer,config.data,self.processor)
            trainer._create_dataloader(train,val,collate_fn,None,None)
            self.assertEqual(len(trainer.train_dataloader),16)
            self.assertEqual(len(trainer.val_dataloader),4)
            for loader,expected in [(trainer.train_dataloader,128),(trainer.val_dataloader,32)]:
                batches=list(loader)
                self.assertEqual(sum(len(batch['input_ids']) for batch in batches),expected)
                self.assertTrue(all(len(batch['input_ids'])==8 for batch in batches))

    def test_monitor_streams_before_exit_and_reports_failure(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            root=Path(tmp)
            mock=root/'nvidia-smi'
            mock.write_text('#!/bin/sh\nprintf "1, 50, 10, 20\\n"\n')
            mock.chmod(0o755)
            env=os.environ.copy()
            env['PATH']=str(root)+os.pathsep+env['PATH']
            env['PYTHONPATH']=str(HERE)
            env['HBA_RUN_DIR']=str(root)
            child='import time; from recording import phase; phase("train",1); time.sleep(1); phase("evaluation",1); time.sleep(0.3); raise SystemExit(7)'
            command=[sys.executable,str(HERE/'resource_monitor.py'),'--gpus','1','--output',str(root/'resources.csv'),
                     '--log',str(root/'run.log'),'--interval','0.1','--',sys.executable,'-c',child]
            process=subprocess.Popen(command,env=env)
            import time
            deadline=time.monotonic()+5
            while not (root/'resources.csv').exists() and time.monotonic()<deadline:
                time.sleep(.05)
            time.sleep(.25)
            self.assertIsNone(process.poll())
            self.assertGreater((root/'resources.csv').stat().st_size,0)
            self.assertEqual(process.wait(timeout=15),7)
            summary=json.loads((root/'resources.summary.json').read_text())
            self.assertEqual(summary['return_code'],7)
            self.assertGreater(summary['phases']['train']['energy_wh'],0)
            self.assertGreater(summary['phases']['evaluation']['wall_time_s'],0)

if __name__=='__main__':
    unittest.main(verbosity=2)
