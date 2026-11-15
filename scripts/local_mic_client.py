import asyncio
import websockets
import pyaudio
import wave
import io
import threading
import tempfile
import os

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
is_recording = False

def capture_audio_to_wav():
    global is_recording
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    frames = []
    
    print("\n🎙️ Recording... (Press ENTER again to stop)")
    while is_recording:
        frames.append(stream.read(CHUNK, exception_on_overflow=False))
        
    stream.stop_stream()
    stream.close()
    
    # Wrap raw PCM frames in a valid WAV header
    wav_io = io.BytesIO()
    with wave.open(wav_io, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
    
    p.terminate()
    return wav_io.getvalue()

def wait_for_enter():
    global is_recording
    input("Press ENTER to start speaking...")
    is_recording = True
    input() # Wait for second Enter
    is_recording = False

async def run_local_call():
    uri = "ws://127.0.0.1:8000/voice/stream/local_test"
    
    # Use a generic context for local testing
    context_id = "test_user_123"
    
    async with websockets.connect(f"{uri}?context_id={context_id}") as websocket:
        print("✅ Connected to FastAPI Voice Server")
        
        while True:
            global is_recording
            
            # Use thread to handle push-to-talk cleanly
            input_thread = threading.Thread(target=wait_for_enter)
            input_thread.start()
            
            while not is_recording:
                await asyncio.sleep(0.1)
                
            wav_bytes = await asyncio.to_thread(capture_audio_to_wav)
            input_thread.join()
            
            print("⏳ Sending valid WAV audio to server...")
            await websocket.send(wav_bytes)
            
            print("⏳ Waiting for AI text and audio response...")
            response_audio = await websocket.recv()
            
            if response_audio:
                print("🔊 Playing response...")
                import pygame
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_mp3:
                    temp_mp3.write(response_audio)
                    temp_path = temp_mp3.name
                
                # Play the MP3 silently in the background
                pygame.mixer.init()
                pygame.mixer.music.load(temp_path)
                pygame.mixer.music.play()
                
                # Wait for the audio to finish playing before continuing
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
                
                pygame.mixer.quit()

if __name__ == "__main__":
    try:
        asyncio.run(run_local_call())
    except KeyboardInterrupt:
        print("\nDisconnected cleanly.")