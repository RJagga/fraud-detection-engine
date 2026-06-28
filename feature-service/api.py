from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import httpx
from features import compute_features, store_transaction

app = FastAPI(title="Feature Engineering Service")

ML_SCORE_URL = "http://localhost:8001/score"

class RawTransaction(BaseModel):
    txn_id:            str
    user_id:           str
    amount:            float
    timestamp:         str
    city:              str
    lat:               float
    lon:               float
    device_id:         str
    merchant_category: str
    is_fraud:          Optional[int] = None   # ground truth label (for testing only)

class EnrichedDecision(BaseModel):
    txn_id:           str
    user_id:          str
    amount:           float
    features:         dict
    risk_score:       float
    risk_label:       str
    decision:         str
    explanation:      dict
    ground_truth:     Optional[int] = None

@app.post("/process", response_model=EnrichedDecision)
async def process_transaction(txn: RawTransaction):
    txn_dict = txn.model_dump()

    # 1. Compute features from Redis history
    features = compute_features(txn_dict)

    # 2. Store this transaction in Redis for future lookups
    store_transaction(txn_dict)

    # 3. Build the payload for the ML scoring service
    #    The ML model was trained on V1-V28 + Amount_scaled from Kaggle.
    #    For our synthetic pipeline we pass the engineered features
    #    mapped to V-columns as a stand-in until Phase 3 retraining.
    #    In production these would be real PCA components.
    import numpy as np
    v_features = {f'V{i}': 0.0 for i in range(1, 29)}
    v_features.update({
        'V1':  float(np.clip(features['txn_velocity_1h'] / 10, -3, 3)),
        'V2':  float(np.clip(features['amount_delta'], -3, 3)),
        'V3':  float(np.clip(features['geo_distance_km'] / 500, -3, 3)),
        'V4':  float(features['new_device']),
        'V5':  float(features['high_risk_merchant']),
        'V6':  float(features['is_night']),
    })

    score_payload = {
        'txn_id': txn.txn_id,
        'amount': txn.amount,
        **v_features
    }

    # 4. Call ML scoring service
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(ML_SCORE_URL, json=score_payload, timeout=5.0)
            resp.raise_for_status()
            score_result = resp.json()
        except httpx.RequestError:
            raise HTTPException(503, "ML scoring service unreachable")

    return EnrichedDecision(
        txn_id       = txn.txn_id,
        user_id      = txn.user_id,
        amount       = txn.amount,
        features     = features,
        risk_score   = score_result['risk_score'],
        risk_label   = score_result['risk_label'],
        decision     = score_result['decision'],
        explanation  = score_result['explanation'],
        ground_truth = txn.is_fraud,
    )

@app.get("/health")
def health():
    return {"status": "ok", "service": "feature-engineering"}