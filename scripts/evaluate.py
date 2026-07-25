import json
from lora_pipeline.core import compare
cases=json.load(open('data/benchmark.json',encoding='utf-8')); base=tuned=0
for x in cases:
 r=compare(x['input'],x['output']); base+=r['scores']['base']; tuned+=r['scores']['fine_tuned']
print({'cases':len(cases),'base_overlap':round(base/len(cases),3),'fine_tuned_overlap':round(tuned/len(cases),3)})
