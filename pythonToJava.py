import requests
from datetime import datetime
import os
from dotenv import load_dotenv
import sys

# .env 파일 로드
load_dotenv(dotenv_path='SilverLink-AI/.env')

BASE_URL = os.getenv('SPRING_BOOT_URL', 'http://localhost:8080')
API_PREFIX = "/api/internal/callbot"

# 환경 변수에서 로그인 정보 가져오기 (없으면 기본값 사용)
ADMIN_ID = os.getenv('ADMIN_ID', 'admin01') 
ADMIN_PW = os.getenv('ADMIN_PW', 'admin01')

# 전역 헤더 (토큰 포함용)
HEADERS = {
    "Content-Type": "application/json"
}

def login():
    """로그인하여 Access Token을 발급받고 헤더에 설정"""
    url = f"{BASE_URL}/api/auth/login"
    data = {
        "loginId": ADMIN_ID,
        "password": ADMIN_PW
    }
    
    try:
        print(f"Logging in to {url} with user '{ADMIN_ID}'...")
        response = requests.post(url, json=data)
        
        if response.status_code == 200:
            token_data = response.json()
            access_token = token_data.get('accessToken')
            if access_token:
                HEADERS["Authorization"] = f"Bearer {access_token}"
                print(HEADERS)
                print("✅ Login Successful! Token acquired.")
                return True
            else:
                print("❌ Login Failed: No access token in response.")
                return False
        else:
            print(f"❌ Login Failed: Status {response.status_code}, Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Login Error: {e}")
        return False

def start_call(elderly_id, name, phone_number):
    """통화 시작 API 호출"""
    url = f"{BASE_URL}{API_PREFIX}/calls"
    data = {
        "elderlyId": elderly_id,
        "name": name,
        "phoneNumber": phone_number,
        "callAt": datetime.now().isoformat()
    }
    response = requests.post(url, json=data, headers=HEADERS)
    return response.json()

def save_prompt(call_id, prompt_text):
    """LLM Prompt 저장 API 호출"""
    url = f"{BASE_URL}{API_PREFIX}/calls/{call_id}/llm/prompt"
    data = {"prompt": prompt_text}
    response = requests.post(url, json=data, headers=HEADERS)
    return response.json()

def save_reply(call_id, content, danger=False):
    """어르신 응답 저장 API 호출"""
    url = f"{BASE_URL}{API_PREFIX}/calls/{call_id}/llm/reply"
    data = {
        "content": content,
        "danger": danger
    }
    response = requests.post(url, json=data, headers=HEADERS)
    return response.json()

def save_message(call_id, speaker, content, danger=False, danger_reason=None, llm_model_id=None):
    """대화 메시지 저장 API 호출 (CALLBOT 또는 ELDERLY)"""
    url = f"{BASE_URL}{API_PREFIX}/calls/{call_id}/messages"
    data = {
        "speaker": speaker,
        "content": content,
        "timestamp": datetime.now().isoformat(),
        "danger": danger,
        "dangerReason": danger_reason,
        "llmModelId": llm_model_id
    }
    response = requests.post(url, json=data, headers=HEADERS)
    return response.json()

def save_summary(call_id, summary_content):
    """통화 요약 저장 API 호출"""
    url = f"{BASE_URL}{API_PREFIX}/calls/{call_id}/summary"
    data = {"content": summary_content}
    response = requests.post(url, json=data, headers=HEADERS)
    return response.json()

def save_emotion(call_id, emotion_level):
    """감정 분석 저장 API 호출 (GOOD, NORMAL, BAD, DEPRESSED)"""
    url = f"{BASE_URL}{API_PREFIX}/calls/{call_id}/emotion"
    data = {"emotionLevel": emotion_level}
    response = requests.post(url, json=data, headers=HEADERS)
    return response.json()

def save_daily_status(call_id, meal_taken, health_status, health_detail, sleep_status, sleep_detail):
    """일일 상태 저장 API 호출"""
    url = f"{BASE_URL}{API_PREFIX}/calls/{call_id}/daily-status"
    data = {
        "mealTaken": meal_taken,
        "healthStatus": health_status,
        "healthDetail": health_detail,
        "sleepStatus": sleep_status,
        "sleepDetail": sleep_detail
    }
    response = requests.post(url, json=data, headers=HEADERS)
    return response.json()

def end_call(call_id, call_time_sec, recording_url=None, summary=None, emotion=None, daily_status=None):
    """통화 종료 API 호출"""
    url = f"{BASE_URL}{API_PREFIX}/calls/{call_id}/end"
    data = {
        "callTimeSec": call_time_sec,
        "recordingUrl": recording_url,
        "summary": summary,
        "emotion": emotion,
        "dailyStatus": daily_status
    }
    response = requests.post(url, json=data, headers=HEADERS)
    return response.json()

if __name__ == "__main__":
    print("--- CallBot Internal API Test (With Auth) ---")
    
    # 0. 로그인 시도
    if not login():
        print("Login failed. Exiting.")
        sys.exit(1)

    call_id = None
    
    # 1. 통화 시작 (3번 고정)
    eid = 3
    print(f"Trying elderly_id={eid}...")
    res = start_call(eid, "홍길동", "010-9876-5432")
    
    if res.get('status') == 'success' or res.get('success') is True:
        print(f"✅ Success with elderly_id={eid}")
        print("Start Call Response:", res)
        # 응답 구조에 따라 callId 추출
        # (실제 응답 구조: {"success": true, "data": {"callId": 123}, "error": null})
        if 'data' in res and 'callId' in res['data']:
            call_id = res['data']['callId']
        else:
            # 혹시 data가 바로 callId인 경우 등 예외 처리 (필요시)
            pass
    else:
        print(f"❌ Failed with elderly_id={eid}")
        print("Full Response:", res)
        call_id = None

    if call_id:
        print(f"--- Proceeding with Call ID: {call_id} ---")

        # 2. LLM Prompt 저장
        prompt_res = save_prompt(call_id, "오늘 기분은 어떠신가요?")
        print("Save Prompt Response:", prompt_res)

        # 3. 어르신 응답 저장
        reply_res = save_reply(call_id, "아주 좋아요.", danger=False)
        print("Save Reply Response:", reply_res)

        # 4. 메시지 저장 (CALLBOT)
        msg_bot = save_message(call_id, "CALLBOT", "다행이네요. 식사는 하셨나요?")
        print("Message (Bot) Response:", msg_bot)

        # 5. 메시지 저장 (ELDERLY)
        msg_user = save_message(call_id, "ELDERLY", "네, 먹었어요.", danger=False)
        print("Message (User) Response:", msg_user)

        # 6. 감정 분석 저장
        emo_res = save_emotion(call_id, "GOOD")
        print("Save Emotion Response:", emo_res)

        # 7. 일일 상태 저장
        daily_res = save_daily_status(call_id, True, "GOOD", "건강함", "GOOD", "숙면")
        print("Save Daily Status Response:", daily_res)

        # 8. 통화 요약 저장
        sum_res = save_summary(call_id, "어르신이 기분이 좋고 식사도 하셨음.")
        print("Save Summary Response:", sum_res)

        # 9. 통화 종료
        end_res = end_call(call_id, 180, "http://example.com/rec.mp3", 
                           summary={"content": "최종 요약"},
                           emotion={"emotionLevel": "GOOD"},
                           daily_status={
                               "mealTaken": True,
                               "healthStatus": "GOOD", 
                               "healthDetail": "이상 무",
                               "sleepStatus": "GOOD",
                               "sleepDetail": "7시간 수면"
                           })
        print("End Call Response:", end_res)
    else:
        print("❌ Could not start call with any tested elderly_id.")