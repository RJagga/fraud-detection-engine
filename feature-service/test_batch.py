import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import httpx, json
from synthetic_data import generate_transaction_stream

FEATURE_SERVICE_URL = "http://localhost:8002/process"

def run_batch(n=50):
    txns    = generate_transaction_stream(n=n, fraud_rate=0.1)
    results = []

    for txn in txns:
        resp = httpx.post(FEATURE_SERVICE_URL, json=txn, timeout=10)
        if resp.status_code == 200:
            result = resp.json()
            results.append(result)
            label  = "🚨 FRAUD" if result['ground_truth'] == 1 else "✅ LEGIT"
            caught = "⚠️  FLAGGED" if result['decision'] != 'APPROVE' else "   passed"
            print(f"{caught} | {label} | score={result['risk_score']:.3f} | "
                  f"vel={result['features']['txn_velocity_1h']} | "
                  f"geo={result['features']['geo_distance_km']:.0f}km | "
                  f"₹{txn['amount']:,.0f}")
        else:
            print(f"Error: {resp.status_code} — {resp.text}")

    # Summary
    flagged = [r for r in results if r['decision'] != 'APPROVE']
    frauds  = [r for r in results if r['ground_truth'] == 1]
    caught  = [r for r in flagged if r['ground_truth'] == 1]

    print(f"\n--- Batch summary ({n} transactions) ---")
    print(f"Actual fraud:   {len(frauds)}")
    print(f"Flagged/Blocked:{len(flagged)}")
    print(f"Fraud caught:   {len(caught)} / {len(frauds)}")

if __name__ == '__main__':
    run_batch(50)