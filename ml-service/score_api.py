from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from explainer import explain_prediction

app = FastAPI(title="Fraud Detection Scoring API")

# Load artifacts once at startup
model     = joblib.load('artifacts/xgboost_model.pkl')
scaler    = joblib.load('artifacts/scaler.pkl')
config    = joblib.load('artifacts/config.pkl')
THRESHOLD = float(config['threshold'])

FEATURE_NAMES = [f'V{i}' for i in range(1, 29)] + ['Amount_scaled']

class TransactionRequest(BaseModel):
    txn_id:  str
    amount:  float
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float

class ScoreResponse(BaseModel):
    txn_id:      str
    risk_score:  float
    risk_label:  str          # LOW / MEDIUM / HIGH
    decision:    str          # APPROVE / FLAG / BLOCK
    explanation: dict

@app.post("/score", response_model=ScoreResponse)
def score_transaction(txn: TransactionRequest):
    # Scale amount the same way as training
    amount_scaled = float(scaler.transform([[txn.amount]])[0][0])

    features = {f'V{i}': getattr(txn, f'V{i}') for i in range(1, 29)}
    features['Amount_scaled'] = amount_scaled

    X = pd.DataFrame([features])[FEATURE_NAMES]
    risk_score = float(model.predict_proba(X)[0][1])

    # Decision logic
    if risk_score >= THRESHOLD:
        decision   = "BLOCK"
        risk_label = "HIGH"
    elif risk_score >= THRESHOLD * 0.6:
        decision   = "FLAG"
        risk_label = "MEDIUM"
    else:
        decision   = "APPROVE"
        risk_label = "LOW"

    explanation = explain_prediction(features, FEATURE_NAMES)

    return ScoreResponse(
        txn_id=txn.txn_id,
        risk_score=round(risk_score, 4),
        risk_label=risk_label,
        decision=decision,
        explanation=explanation
    )

@app.get("/health")
def health():
    return {"status": "ok", "threshold": float(THRESHOLD)}