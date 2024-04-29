import asyncio
import websockets
import json
import time

async def websocket_client():
    uri = "ws://8.138.92.240:8765"
    async with websockets.connect(uri) as websocket:
        await websocket.send('{"client_type":"receiver"}')  # Identify as receiver
        while True:
            response = await websocket.recv()
            print(f"Received from server: {response}")

asyncio.run(websocket_client())