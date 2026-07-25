import argparse,json
from .core import compare,load_jsonl,split
from .train import train

def main():
 p=argparse.ArgumentParser(); s=p.add_subparsers(dest='cmd',required=True)
 v=s.add_parser('serve'); v.add_argument('--host',default='127.0.0.1'); v.add_argument('--port',type=int,default=8610)
 t=s.add_parser('train'); t.add_argument('--config',default='config/train.yaml'); t.add_argument('--dry-run',action='store_true')
 c=s.add_parser('compare'); c.add_argument('prompt')
 d=s.add_parser('validate'); d.add_argument('--dataset',default='data/support_tone.jsonl')
 a=p.parse_args()
 if a.cmd=='serve': import uvicorn; uvicorn.run('lora_pipeline.api:app',host=a.host,port=a.port)
 elif a.cmd=='train': print(json.dumps(train(a.config,a.dry_run),indent=2))
 elif a.cmd=='compare': print(json.dumps(compare(a.prompt),indent=2))
 else: print({k:len(v) for k,v in split(load_jsonl(a.dataset)).items()})
