import asyncio
import pdb

import websockets
import json
import time
from SMPLVisualizer import SMPLVisualizer
import torch

async def websocket_client():
    uri = "ws://8.138.92.240:8765"
    async with websockets.connect(uri) as websocket:
        await websocket.send('{"client_type":"receiver"}')  # Identify as receiver
        SV = SMPLVisualizer()
        while True:
            response = await websocket.recv()
            data = json.loads(response)
            pose = data['pose']
            trans = data['trans']
            # Assuming pose and trans are lists
            pose = torch.tensor(pose)
            trans = torch.tensor(trans)

            SV.visualize_smpl_with_tensors(pose, trans)

asyncio.run(websocket_client())