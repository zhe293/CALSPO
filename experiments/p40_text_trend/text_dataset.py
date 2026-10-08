"""Frozen pure-text HBA dataset; read-only sources, fixed middle truncation."""
import copy
import hashlib
from pathlib import Path
import pyarrow.parquet as pq
import torch
from jinja2 import Template
from torch.utils.data import Dataset
from verl.utils.torch_functional import postprocess_data

class TextTrendDataset(Dataset):
    def __init__(self, data_files, tokenizer, config, processor=None):
        files = [data_files] if isinstance(data_files, (str, Path)) else list(data_files)
        self.rows = [row for f in files for row in pq.read_table(f).to_pylist()]
        self.tokenizer, self.processor, self.config = tokenizer, processor, config
        self.template = Template(Path(config.format_prompt).read_text(encoding='utf-8'))
        self.max_length = int(config.max_prompt_length)
        if not self.rows:
            raise ValueError('Empty dataset')
        for row in self.rows:
            if any(row.get(k) for k in ('images', 'videos', 'audios')):
                raise ValueError('Pure text required')
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, index):
        row = copy.deepcopy(self.rows[index])
        content = self.template.render(content=str(row['problem']))
        if self.processor is not None:
            prompt = self.processor.apply_chat_template(
                [{'role':'user','content':[{'type':'text','text':content}]}], tokenize=False, add_generation_prompt=True)
            if isinstance(prompt,list):
                if len(prompt)!=1:
                    raise ValueError('Expected one prompt')
                prompt=prompt[0]
        else:
            prompt=self.tokenizer.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True)
        ids=self.tokenizer.encode(prompt,add_special_tokens=False)
        original_length=len(ids)
        if len(ids)>self.max_length:
            left=self.max_length//2
            ids=ids[:left]+ids[-(self.max_length-left):]
        inputs,mask=postprocess_data(input_ids=torch.tensor([ids]),attention_mask=torch.ones(1,len(ids),dtype=torch.long),
                                    max_length=self.max_length,pad_token_id=self.tokenizer.pad_token_id,left_pad=True,truncation='error')
        return {'input_ids':inputs[0], 'attention_mask':mask[0], 'position_ids':(mask.cumsum(-1)-1).clamp(min=0)[0],
                'raw_prompt_ids':ids,'reward_model':{'ground_truth':str(row['answer'])},'data_source':str(row['task']),
                'task':str(row['task']),'dataset':str(row['dataset']),'class_label':str(row.get('class_label',row['answer'])),
                'modality_signature':'text','index':index,
                'extra_info':{'index':index,'original_prompt_tokens':original_length,'truncated':original_length>self.max_length,
                              'problem_sha256':hashlib.sha256(str(row['problem']).encode()).hexdigest()}}
