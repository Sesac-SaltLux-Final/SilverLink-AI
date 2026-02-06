import requests
import httpx
from typing import Optional

class TTS:
    def __init__(self, api_key: str, url: str) -> None:
        self.api_key = api_key
        self.url = url
        self.client = httpx.AsyncClient(timeout=10.0)
        self.cache = {}
        
    async def close(self):
        await self.client.aclose()
        
    def sultlux(self, text: str):
        """동기 방식 Saltlux TTS 호출"""
        if text in self.cache:
            return self.cache[text]

        headers = {
            "apikey": self.api_key,
            "Content-Type": "application/json"
        }
        payload = {"input": text, "voice": 29, "lang": "ko"}

        # self.url이 올바른 엔드포인트여야 합니다.
        response = requests.post(self.url, headers=headers, json=payload)
        response.raise_for_status()
        
        self.cache[text] = response.content
        return response.content

    async def asultlux(self, text: str) -> Optional[bytes]:
        """비동기 방식 Saltlux TTS 호출 (폴백 제거됨)"""
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
                print(f"❌ Saltlux TTS API Error: {resp.status_code} - {resp.text}")
                return None
        except Exception as e:
            print(f"❌ Saltlux TTS Network Error: {e}")
            return None