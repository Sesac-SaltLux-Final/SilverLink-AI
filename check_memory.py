import requests
import sys

def check_memory_via_api(elderly_id):
    # 서버 주소 (FastAPI 기본 포트 5000)
    url = f"http://localhost:5000/api/callbot/memory/{elderly_id}"
    
    print(f"🔍 [장기 기억 조회] API 호출 중: {url}")
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            memories = data.get("memories", [])
            
            if not memories:
                print(f"📭 어르신 ID {elderly_id}에 대한 저장된 기억이 없습니다.")
                return

            print(f"✅ 어르신 ID {elderly_id}의 기억을 찾았습니다.\n")
            print("=" * 60)
            
            # Mem0 응답이 {'results': [...]} 형태인 경우 대응
            actual_memories = memories
            if isinstance(memories, dict) and "results" in memories:
                actual_memories = memories["results"]
            elif isinstance(memories, list) and len(memories) > 0 and isinstance(memories[0], dict) and "results" in memories[0]:
                actual_memories = memories[0]["results"]

            for m in actual_memories:
                if isinstance(m, dict):
                    content = m.get('memory', m.get('data', '내용 없음'))
                    timestamp = m.get('created_at', m.get('updated_at', ''))
                    print(f"  • {content} ({timestamp})")
                else:
                    print(f"  • {str(m)}")
            print("=" * 60)
        elif response.status_code == 404:
            print("❌ 에러: API 엔드포인트를 찾을 수 없습니다. (404 Not Found)")
            print("💡 서버가 정상적으로 실행 중인지, 주소가 '/api/callbot/memory/...' 가 맞는지 확인하세요.")
        else:
            print(f"❌ 에러 발생 (상태 코드: {response.status_code})")
            print(response.text)
            
    except Exception as e:
        print(f"❌ 서버 접속 실패: {e}")
        print("💡 서버(uvicorn)가 켜져 있는지 확인해 주세요.")

if __name__ == "__main__":
    # 조회하고 싶은 어르신 ID를 입력 (기본값 695)
    eid = sys.argv[1] if len(sys.argv) > 1 else 3
    check_memory_via_api(eid)
