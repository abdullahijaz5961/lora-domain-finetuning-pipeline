from lora_pipeline.core import load_jsonl,split,compare

def test_dataset_and_split():
 rows=load_jsonl('data/support_tone.jsonl'); parts=split(rows); assert len(rows)==600; assert sum(map(len,parts.values()))==600

def test_comparison():
 r=compare('I was charged twice'); assert 'billing' in r['fine_tuned'].lower()
