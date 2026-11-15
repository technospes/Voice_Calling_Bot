import asyncio
import websockets
import json

async def simulate_call():
    uri = "ws://127.0.0.1:8000/voice/stream/sim_999"
    
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected to server!")
            
            # Wait for greeting
            greeting = await websocket.recv()
            print(f"Agent says: {json.loads(greeting)['text']}")
            
            # Simulate pressing '1'
            print("\nPressing '1'...")
            await websocket.send(json.dumps({"event": "dtmf", "digit": "1"}))
            
            response1 = await websocket.recv()
            print(f"Agent says: {json.loads(response1)['text']}")

            # Simulate speaking to the agent (Echo Test)
            print("\nSpeaking: 'Mujhe do paneer tikka chahiye'...")
            await websocket.send(json.dumps({"event": "speech", "text": "Mujhe do paneer tikka chahiye"}))
            
            echo_response = await websocket.recv()
            print(f"Agent says: {json.loads(echo_response)['text']}")

            # Simulate confirming the order (triggering the database save)
            print("\nSpeaking: 'Bas aur kuch nahi chahiye. Order done.'...")
            await websocket.send(json.dumps({"event": "speech", "text": "Bas aur kuch nahi chahiye. Order done."}))
            
            final_response = await websocket.recv()
            print(f"Agent says: {json.loads(final_response)['text']}")

    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    asyncio.run(simulate_call())