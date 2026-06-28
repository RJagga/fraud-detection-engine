import redis
import json
import math
from datetime import datetime, timedelta

r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

HISTORY_WINDOW_SECONDS = 3600   # 1 hour velocity window
MAX_HISTORY_PER_USER   = 50     # keep last 50 txns in Redis per user

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2 +
         math.cos(math.radians(lat1)) *
         math.cos(math.radians(lat2)) *
         math.sin(dlon/2)**2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def get_user_history(user_id: str) -> list:
    """Fetch user's recent transactions from Redis sorted set."""
    key = f"user_history:{user_id}"
    raw = r.zrange(key, 0, -1, withscores=True)
    return [
        {'data': json.loads(item), 'timestamp': score}
        for item, score in raw
    ]

def store_transaction(txn: dict):
    """Push transaction into Redis sorted set (score = unix timestamp)."""
    key       = f"user_history:{txn['user_id']}"
    ts        = datetime.fromisoformat(txn['timestamp']).timestamp()
    r.zadd(key, {json.dumps(txn): ts})
    r.zremrangebyrank(key, 0, -(MAX_HISTORY_PER_USER + 1))  # trim oldest
    r.expire(key, 86400 * 7)                                  # TTL: 7 days

def compute_features(txn: dict) -> dict:
    """
    Given a raw transaction, pull user history from Redis
    and compute all engineered features.
    """
    history  = get_user_history(txn['user_id'])
    now_ts   = datetime.fromisoformat(txn['timestamp']).timestamp()
    window   = now_ts - HISTORY_WINDOW_SECONDS

    # --- Velocity: txn count in last 60 minutes ---
    recent = [h for h in history if h['timestamp'] >= window]
    txn_velocity_1h = len(recent)

    # --- Amount deviation from user's historical mean ---
    all_amounts = [h['data']['amount'] for h in history]
    if all_amounts:
        user_avg    = sum(all_amounts) / len(all_amounts)
        amount_delta = (txn['amount'] - user_avg) / (user_avg + 1)
    else:
        user_avg     = txn['amount']
        amount_delta = 0.0

    # --- Geo delta: distance from last transaction ---
    if history:
        last = sorted(history, key=lambda x: x['timestamp'])[-1]['data']
        geo_distance_km = haversine_km(
            last['lat'], last['lon'],
            txn['lat'],  txn['lon']
        )
    else:
        geo_distance_km = 0.0

    # --- Device: has this device been seen before for this user? ---
    seen_devices = {h['data']['device_id'] for h in history}
    new_device   = int(txn['device_id'] not in seen_devices)

    # --- Merchant risk ---
    high_risk_merchant = int(txn['merchant_category'] in ['crypto', 'gaming'])

    # --- Time of day ---
    hour     = datetime.fromisoformat(txn['timestamp']).hour
    is_night = int(0 <= hour <= 5)

    return {
        'txn_id':              txn['txn_id'],
        'user_id':             txn['user_id'],
        'amount':              txn['amount'],
        'txn_velocity_1h':     txn_velocity_1h,
        'amount_delta':        round(amount_delta, 4),
        'geo_distance_km':     round(geo_distance_km, 2),
        'new_device':          new_device,
        'high_risk_merchant':  high_risk_merchant,
        'is_night':            is_night,
        'user_avg_amount':     round(user_avg, 2),
        'history_length':      len(history),
    }