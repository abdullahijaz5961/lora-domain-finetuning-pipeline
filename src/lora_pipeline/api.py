from fastapi import FastAPI
from pydantic import BaseModel
from .core import compare,health_summary
app=FastAPI(title='LoRA Fine-Tuning A/B API')
class Compare(BaseModel): prompt:str; expected:str=''
@app.get('/health')
def health():return health_summary()
@app.post('/v1/compare')
def ab(x:Compare):return compare(x.prompt,x.expected)
