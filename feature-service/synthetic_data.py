import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from faker import Faker
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

fake = Faker('en_IN')

CITY_COORDS = {
    'Mumbai':    (19.07, 72.87),
    'Delhi':     (28.61, 77.21),
    'Bengaluru': (12.97, 77.59),
    'Chennai':   (13.08, 80.27),
    'Hyderabad': (17.38, 78.48),
    'Kolkata':   (22.57, 88.36),
}

MERCHANT_CATEGORIES = ['grocery', 'electronics', 'travel', 'gaming', 'crypto', 'food', 'clothing']

def generate_user_pool(n_users=200):
    """Fixed pool of users with stable device IDs."""
    return [
        {
            'user_id':   fake.uuid4(),
            'device_id': fake.uuid4(),
            'home_city': random.choice(list(CITY_COORDS.keys())),
        }
        for _ in range(n_users)
    ]

def generate_transaction(user: dict, timestamp: datetime, is_fraud=False) -> dict:
    city = random.choice(list(CITY_COORDS.keys()))
    if not is_fraud:
        city = user['home_city']  # legit txns stay near home city

    lat, lon = CITY_COORDS[city]

    # Fraud patterns: new device, far city, large amount, late night
    device_id = fake.uuid4() if is_fraud and random.random() < 0.7 else user['device_id']
    amount    = round(np.random.uniform(5000, 50000), 2) if is_fraud else round(np.random.lognormal(6, 1), 2)
    hour      = random.randint(0, 4) if is_fraud else random.randint(8, 22)
    timestamp = timestamp.replace(hour=hour)

    return {
        'txn_id':            fake.uuid4(),
        'user_id':           user['user_id'],
        'amount':            amount,
        'timestamp':         timestamp.isoformat(),
        'city':              city,
        'lat':               lat + np.random.normal(0, 0.02),
        'lon':               lon + np.random.normal(0, 0.02),
        'device_id':         device_id,
        'merchant_category': random.choice(['crypto', 'gaming']) if is_fraud
                             else random.choice(MERCHANT_CATEGORIES),
        'is_fraud':          int(is_fraud),
    }

def generate_transaction_stream(n=500, fraud_rate=0.05):
    """Generate a stream of transactions with realistic fraud rate."""
    users = generate_user_pool()
    base  = datetime.now() - timedelta(hours=6)
    txns  = []

    for i in range(n):
        user      = random.choice(users)
        timestamp = base + timedelta(seconds=i * 10)
        is_fraud  = random.random() < fraud_rate
        txns.append(generate_transaction(user, timestamp, is_fraud))

    return txns