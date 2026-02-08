import httpx
import os
import io
import wave
import audioop
import urllib.parse
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
import uvicorn

load_dotenv()

app = FastAPI()

# Configuration (환경변수에서 로드)
NGROK_URL = os.getenv("CALL_CONTROLL_URL", "http://localhost:5000")
LUXIA_API_KEY = os.getenv("LUXIA_API_KEY", "")

# OpenAI Client
aclient = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Global HTTP Client (Connection Pooling for Speed)
http_client = httpx.AsyncClient(timeout=10.0)

# History
history = [
    {"role": "system", "content": "You are a helpful voice assistant. Keep your answers concise (1-2 sentences) and suitable for speech synthesis."}
]

@app.on_event("shutdown")
async def shutdown_event():
    await http_client.aclose()

# --- Audio Utils ---
def wav_to_ulaw(wav_bytes: bytes) -> bytes:
    """Converts WAV bytes to raw Mu-law audio (8kHz, Mono) without headers"""
    try:
        with wave.open(io.BytesIO(wav_bytes), 'rb') as wav:
            # Resample and Convert
            n_channels = wav.getnchannels()
            framerate = wav.getframerate()
            sampwidth = wav.getsampwidth()
            n_frames = wav.getnframes()
            data = wav.readframes(n_frames)

            # Ensure Mono
            if n_channels > 1:
                data = audioop.tomono(data, sampwidth, 0.5, 0.5)
            
            # Ensure 8000Hz
            if framerate != 8000:
                data, _ = audioop.ratecv(data, sampwidth, 1, framerate, 8000, None)

            # Linear PCM -> Mu-law
            return audioop.lin2ulaw(data, sampwidth)
    except Exception as e:
        print(f"Audio Conv Error: {e}")
        return b""

async def generate_tts_stream(text: str):
    """Calls Luxia TTS and returns raw WAV bytes"""
    url = "https://bridge.luxiacloud.com/luxia/v1/text-to-speech"
    headers = {"apikey": LUXIA_API_KEY, "Content-Type": "application/json"}
    payload = {"input": text, "voice": 29, "lang": "ko"}

    try:
        resp = await http_client.post(url, headers=headers, json=payload)
        if resp.status_code == 200:
            return resp.content
    except Exception as e:
        print(f"TTS Error: {e}")
    return None

# --- Streaming Logic ---
async def ai_response_generator(user_text: str):
    """
    The Core Pipeline: LLM Stream -> Text Buffer -> TTS -> Audio Stream
    """
    global history
    history.append({"role": "user", "content": user_text})
    
    # Send some silence first to prime the buffer (0.2s)
    yield b'\xff' * 1600 

    full_response = ""
    buffer = ""
    
    try:
        # LLM Request (gpt-4o-mini for speed)
        stream = await aclient.chat.completions.create(
            model="gpt-4o-mini",
            messages=history,
            stream=True
        )

        async for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                full_response += content
                buffer += content
                
                # ⚡ Aggressive Chunking: Trigger TTS on punctuation OR length > 10
                # This makes the first audio packet arrive much faster.
                is_punct = any(p in content for p in [".", "?", "!", ",", "\n"])
                is_long = len(buffer) > 10 and content.endswith(" ")
                
                if is_punct or is_long:
                    # Generate TTS for this chunk
                    print(f"🗣️ TTS Chunk: {buffer}")
                    wav_data = await generate_tts_stream(buffer)
                    if wav_data:
                        ulaw_data = wav_to_ulaw(wav_data)
                        yield ulaw_data # Stream audio to Twilio
                    buffer = ""

        # Process remaining buffer
        if buffer.strip():
            wav_data = await generate_tts_stream(buffer)
            if wav_data:
                yield wav_to_ulaw(wav_data)

        history.append({"role": "assistant", "content": full_response})
        print(f"🤖 Full AI Response: {full_response}")

    except Exception as e:
        print(f"Stream Error: {e}")

# --- Routes ---

@app.post("/gather")
async def gather(request: Request):
    """Twilio SpeechResult Handler"""
    form_data = await request.form()
    speech_result = form_data.get("SpeechResult")

    if not speech_result:
        # Retry
        return Response(content="""
        <Response>
            <Gather input="speech" action="/gather" method="POST" language="ko-KR" speechTimeout="auto">
            </Gather>
        </Response>
        """, media_type="application/xml")

    print(f"🎤 User said: {speech_result}")
    
    # We pass the user text to the streaming endpoint via query param
    encoded_text = urllib.parse.quote(speech_result)
    stream_url = f"{NGROK_URL}/stream_response?text={encoded_text}"

    # Return TwiML that plays the stream
    # bargeIn="true" allows the user to interrupt the stream immediately
    twiml = f"""
    <Response>
        <Gather input="speech" action="/gather" method="POST" language="ko-KR" speechTimeout="auto" bargeIn="true">
            <Play contentType="audio/basic">{stream_url}</Play>
        </Gather>
    </Response>
    """
    return Response(content=twiml, media_type="application/xml")

@app.get("/stream_response")
async def stream_response(text: str):
    """Streams audio chunks dynamically generated from LLM -> TTS"""
    return StreamingResponse(
        ai_response_generator(text),
        media_type="audio/basic",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "X-Accel-Buffering": "no" # Disable Nginx buffering if any
        }
    )

@app.post("/voice")
async def voice(request: Request):
    """Initial Call"""
    greeting = "안녕하세요! 무엇을 도와드릴까요?"
    encoded_greeting = urllib.parse.quote(greeting)
    stream_url = f"{NGROK_URL}/stream_response?text={encoded_greeting}"

    twiml = f"""
    <Response>
        <Gather input="speech" action="/gather" method="POST" language="ko-KR" speechTimeout="auto" bargeIn="true">
            <Play contentType="audio/basic">{stream_url}</Play>
        </Gather>
    </Response>
    """
    return Response(content=twiml, media_type="application/xml")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)