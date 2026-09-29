"""
api/websocket.py
----------------
WebSocket endpoints for pushing real-time updates to connected clients.
Listens to Redis pub/sub channels.
"""

import os
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import redis.asyncio as aioredis
import asyncio
import json

router = APIRouter()

@router.websocket("/ws/screener")
async def websocket_screener(websocket: WebSocket):
    await websocket.accept()
    
    # Connect to Redis asynchronously (127.0.0.1 avoids Windows IPv6 delay)
    redis_host = os.getenv("REDIS_HOST", "127.0.0.1")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))
    r = await aioredis.Redis(host=redis_host, port=redis_port, db=0, decode_responses=True)
    pubsub = r.pubsub()
    
    # Subscribe to the screener updates channel
    await pubsub.subscribe("screener:updates")
    
    try:
        while True:
            # Wait for a message from Redis pub/sub
            message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=0.1)
            if message is not None:
                # Forward the message data to the WebSocket client
                data = message['data']
                await websocket.send_text(data)
            
            # Yield control back to the event loop
            await asyncio.sleep(0.01)
            
    except WebSocketDisconnect:
        print("[WebSocket] Client disconnected")
    except Exception as e:
        print(f"[WebSocket Error] {e}")
    finally:
        await pubsub.unsubscribe("screener:updates")
        await pubsub.close()
        await r.aclose()
