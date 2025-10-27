import pickle
from fastapi import FastAPI
from typing import Dict, Any
import uvicorn

app=FastAPI(title="lead-convertion")
with open('pipeline_v1.bin','rb') as f_in:
    pipeline = pickle.load(f_in)

@app.post("/predict")
def predict(lead: Dict[str, Any]):
    converted_prob = float(pipeline.predict_proba(lead)[0,1])
    return{
        "probability of getting subscribed = ":converted_prob,
        "isConverted = ":bool(pipeline.predict(lead))
    }

if __name__== "__main__":
    uvicorn.run(app,host="localhost",port=9696)