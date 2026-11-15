import uuid
import asyncio
import os
import pyaudio
from dotenv import load_dotenv
from livekit import rtc, api

# Load credentials from .env
load_dotenv()

SAMPLE_RATE = 24000
CHANNELS = 1
CHUNK_SAMPLES = int(SAMPLE_RATE * 0.02)  # 20ms audio chunks

async def main():
    url = os.getenv("LIVEKIT_URL")
    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")
    room_name = f"test-order-room-{uuid.uuid4().hex[:8]}"

    print("🚀 Requesting Agent Dispatch from LiveKit Cloud...")
    lk_api = api.LiveKitAPI(url, api_key, api_secret)
    try:
        # This simulates the "Start Session" button in the LiveKit Playground
        await lk_api.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(
                agent_name="order-agent",
                room=room_name
            )
        )
    except Exception as e:
        print(f"⚠️ Dispatch check: {e}")
    finally:
        await lk_api.aclose()

    # Generate a connection token for this local script
    token = api.AccessToken(api_key, api_secret) \
        .with_identity("python-customer") \
        .with_name("Local Test Customer") \
        .with_grants(api.VideoGrants(room_join=True, room=room_name)) \
        .to_jwt()

    print("🔌 Connecting to Room...")
    room = rtc.Room()

    # Initialize Desktop Audio (Speakers and Microphone)
    p = pyaudio.PyAudio()
    in_stream = p.open(format=pyaudio.paInt16, channels=CHANNELS, rate=SAMPLE_RATE, input=True, frames_per_buffer=CHUNK_SAMPLES)

    # (Remove the old out_stream = p.open(...) line before this if you have it)
    
    @room.on("track_subscribed")
    def on_track_subscribed(track: rtc.Track, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            print("🔊 AI Agent Audio Connected! Playing...")
            audio_stream = rtc.AudioStream(track)
            
            async def play_audio():
                out_stream = None
                async for event in audio_stream:
                    # Dynamically configure the speaker using WebRTC's exact frame specs
                    if out_stream is None:
                        out_stream = p.open(
                            format=pyaudio.paInt16,
                            channels=event.frame.num_channels,
                            rate=event.frame.sample_rate,
                            output=True
                        )
                    # Play the audio frame
                    await asyncio.to_thread(out_stream.write, event.frame.data.tobytes())
            
            asyncio.create_task(play_audio())

    await room.connect(url, token)
    print(f"✅ Customer connected to Room: {room_name}")

    # Connect desktop microphone to LiveKit
    source = rtc.AudioSource(SAMPLE_RATE, CHANNELS)
    track = rtc.LocalAudioTrack.create_audio_track("mic", source)
    options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
    await room.local_participant.publish_track(track, options)

    print("\n🎙️ Microphone LIVE. The AI will speak shortly. (Press Ctrl+C to hang up)\n")

    try:
        while True:
            # Read from hardware microphone and push to WebRTC
            data = await asyncio.to_thread(in_stream.read, CHUNK_SAMPLES, exception_on_overflow=False)
            frame = rtc.AudioFrame(
                data=data,
                sample_rate=SAMPLE_RATE,
                num_channels=CHANNELS,
                samples_per_channel=CHUNK_SAMPLES
            )
            await source.capture_frame(frame)
    except KeyboardInterrupt:
        print("\n🛑 Hanging up...")
    finally:
        await room.disconnect()
        in_stream.stop_stream()
        in_stream.close()
        p.terminate()

if __name__ == "__main__":
    asyncio.run(main())