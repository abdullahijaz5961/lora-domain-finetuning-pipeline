from __future__ import annotations
import argparse,json
from pathlib import Path
from .core import load_jsonl,split

def train(config_path='config/train.yaml',dry_run=False):
 import yaml
 cfg=yaml.safe_load(Path(config_path).read_text()); rows=load_jsonl(cfg['dataset']); parts=split(rows,cfg.get('seed',42))
 summary={'base_model':cfg['base_model'],'records':{k:len(v) for k,v in parts.items()},'lora':cfg['lora'],'dry_run':dry_run}
 if dry_run:return summary
 try:
  import torch
  from datasets import Dataset
  from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig,TrainingArguments
  from peft import LoraConfig
  from trl import SFTTrainer
 except ImportError as e: raise RuntimeError('Install training extras: pip install -e ".[training]"') from e
 tokenizer=AutoTokenizer.from_pretrained(cfg['base_model']); tokenizer.pad_token=tokenizer.eos_token
 quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=True)
 model=AutoModelForCausalLM.from_pretrained(cfg['base_model'],quantization_config=quant,device_map='auto')
 def fmt(x):return f"### Instruction\n{x['instruction']}\n### Input\n{x['input']}\n### Response\n{x['output']}"
 ds=Dataset.from_list([{**x,'text':fmt(x)} for x in parts['train']]); val=Dataset.from_list([{**x,'text':fmt(x)} for x in parts['validation']])
 lc=cfg['lora']; peft=LoraConfig(r=lc['rank'],lora_alpha=lc['alpha'],lora_dropout=lc['dropout'],target_modules=lc['targets'],task_type='CAUSAL_LM')
 args=TrainingArguments(output_dir=cfg['output_dir'],num_train_epochs=cfg['epochs'],per_device_train_batch_size=1,gradient_accumulation_steps=8,learning_rate=cfg['learning_rate'],logging_steps=10,save_strategy='epoch',eval_strategy='epoch',bf16=True,report_to=[])
 trainer=SFTTrainer(model=model,tokenizer=tokenizer,train_dataset=ds,eval_dataset=val,peft_config=peft,args=args,dataset_text_field='text',max_seq_length=cfg['max_seq_length'])
 trainer.train(); trainer.model.save_pretrained(Path(cfg['output_dir'])/'adapter'); tokenizer.save_pretrained(Path(cfg['output_dir'])/'adapter'); return summary
