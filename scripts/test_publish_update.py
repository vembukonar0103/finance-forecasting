"""
scripts/test_publish_update.py
------------------------------
Publishes a fake screener update to the Redis pub/sub channel.
"""

import redis
import json
import time

def publish_test_update():
    r = redis.Redis(host='localhost', port=6379, db=0)
    fake_update = {
        "ticker": "AAPL",
        "update_type": "screener_match",
        "composite_rating": {
            "ai_rating": "Strong Bullish",
            "signal_status": "confirmation",
            "kronos_pattern": "CDLENGULFING",
            "timesfm_direction": "Bullish"
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    
    channel = "screener:updates"
    r.publish(channel, json.dumps(fake_update))
    print(f"Published test update to {channel}")

if __name__ == "__main__":
    publish_test_update()
