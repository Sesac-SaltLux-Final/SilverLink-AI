import httpx
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("app.util.http_client")

async def send_data_to_backend(
    url: str, 
    payload: Dict[str, Any], 
    method: str = "POST", 
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 10.0
) -> Optional[Dict[str, Any]]:
    """
    Java(Spring Boot) 백엔드 또는 외부 API로 JSON 데이터를 전송하는 범용 비동기 함수.

    Args:
        url (str): API 엔드포인트 URL (예: "http://localhost:8080/api/v1/result")
        payload (dict): 전송할 데이터 (JSON 형식으로 변환됨)
        method (str): HTTP 메서드 ("POST", "PUT" 등). 기본값은 "POST".
        headers (dict, optional): 추가 헤더. 기본적으로 Content-Type: application/json은 포함됨.
        timeout (float): 요청 타임아웃 (초). 기본값 10초.

    Returns:
        dict: 성공 시 응답 JSON 데이터.
        None: 실패 시 None 반환 (로그에 에러 기록).
    """
    
    if headers is None:
        headers = {}
    
    # 기본 헤더 설정
    if "Content-Type" not in headers:
        headers["Content-Type"] = "application/json"

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            auth_status = "None"
            if headers and "Authorization" in headers:
                token = headers["Authorization"]
                if len(token) > 15:
                    auth_status = f"Present ({token[:15]}...)"
                else:
                    auth_status = "Present (Short)"
            
            logger.info(f"📤 Sending data to {url} | Method: {method} | Auth: {auth_status}")
            
            response = await client.request(
                method=method,
                url=url,
                json=payload,
                headers=headers
            )
            
            response.raise_for_status() # 4xx, 5xx 에러 시 예외 발생
            
            logger.info(f"✅ Data sent successfully. Status: {response.status_code}")
            
            # 응답이 비어있지 않으면 JSON 반환
            if response.content:
                return response.json()
            return {}

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                logger.info(f"🔑 Auth required (401) for {url}. Returning status for automatic retry.")
                return {"error": "UNAUTHORIZED_401"}
            else:
                logger.error(f"❌ HTTP Error: {e.response.status_code} - {e.response.text}")
            return None
        except httpx.RequestError as e:
            logger.error(f"❌ Connection Error sending to {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Unexpected Error: {e}")
            return None
