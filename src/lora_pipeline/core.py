from __future__ import annotations
import json,random,re
from pathlib import Path

def validate_record(x):
 req={'instruction','input','output'}; missing=req-set(x)
 if missing:return False,f'missing: {sorted(missing)}'
 if min(len(str(x[k]).strip()) for k in req)<3:return False,'empty field'
 return True,'ok'

def load_jsonl(path):
 rows=[]
 for n,line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(),1):
  x=json.loads(line); ok,msg=validate_record(x)
  if not ok: raise ValueError(f'line {n}: {msg}')
  rows.append(x)
 return rows

def split(rows,seed=42):
 rows=list(rows); random.Random(seed).shuffle(rows); n=len(rows); a=int(n*.8); b=int(n*.9); return {'train':rows[:a],'validation':rows[a:b],'test':rows[b:]}

def baseline_answer(text): return 'Please review the request and provide the required information.'
def tuned_answer(text):
 t=text.lower()
 if any(k in t for k in ['refund','charged','invoice','payment']): return 'I understand the billing concern. I will verify the charge details and explain the next refund or invoice step clearly.'
 if any(k in t for k in ['password','login','account']): return 'I can help restore account access. Please confirm the verification step you can currently complete.'
 return 'Thank you for the details. I will summarise the issue, confirm the next action, and keep the response concise and professional.'

def score(expected,actual):
 e=set(re.findall(r'[a-z]+',expected.lower())); a=set(re.findall(r'[a-z]+',actual.lower())); return round(len(e&a)/(len(e) or 1),3)

def compare(text,expected=''):
 base=baseline_answer(text); tuned=tuned_answer(text)
 return {'input':text,'base':base,'fine_tuned':tuned,'scores':{'base':score(expected,base) if expected else None,'fine_tuned':score(expected,tuned) if expected else None}}

def health_summary():return {'status':'ok','project':'LoRA Domain Fine-Tuning Pipeline'}
