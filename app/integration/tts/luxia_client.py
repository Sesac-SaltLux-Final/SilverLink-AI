import requests
import httpx

class TTS:
    def __init__(self, api_key: str, url: str) -> None:
        self.api_key = api_key
        self.url = url
        self.client = httpx.AsyncClient(timeout=10.0)
        self.cache = {}
        
    async def close(self):
        await self.client.aclose()
        
    def sultlux(self, text:str):
        if text in self.cache:
            return self.cache[text]

        url = "url"
        headers = {
            "apikey": self.api_key,
            "Content-Type": "application/json"
        }
        payload = {"input": text, "voice": 29, "lang": "ko"}

        # requests is sync, so we keep it separate or use self.client in a sync wrapper
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        
        return response.content

    async def asultlux(self, text: str):
        if text in self.cache:
            print(f"⚡ Using Cached TTS for: {text[:10]}...")
            return self.cache[text]

        url = "https://bridge.luxiacloud.com/luxia/v1/text-to-speech" 
        headers = {
            "apikey": self.api_key,
            "Content-Type": "application/json"
        }
        payload = {"input": text, "voice": 29, "lang": "ko"}

        try:
            resp = await self.client.post(url, headers=headers, json=payload)
            if resp.status_code == 200:
                self.cache[text] = resp.content
                return resp.content
            else:
                print(f"⚠️ TTS API Error: {resp.status_code} - {resp.text}")
                print("🔄 Switching to OpenAI TTS Fallback...")
                return await self._fallback_openai_tts(text)
        except Exception as e:
            print(f"TTS Error: {e}")
            return await self._fallback_openai_tts(text)
        return None

    async def _fallback_openai_tts(self, text: str):
        from openai import AsyncOpenAI
        from app.core.config import configs
        
        try:
            client = AsyncOpenAI(api_key=configs.OPENAI_API_KEY)
            response = await client.audio.speech.create(
                model="tts-1",
                voice="nova",
                input=text,
                response_format="wav"  # Important: request WAV for compatibility
            )
            return response.content
        except Exception as e:
            print(f"❌ Fallback TTS Error: {e}")
            return None
        
        

        