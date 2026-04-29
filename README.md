# SilverLink-AI

**SilverLink-AI**는 LLM 기반의 콜봇(Callbot)과 챗봇(Chatbot)을 통합한 지능형 상담 AI 시스템입니다. 고령자나 돌봄이 필요한 사용자를 대상으로 건강 상태, 식사 여부, 기분 등을 체크하고, 위급 상황 발생 시 즉각적인 대응을 지원합니다.

<br><br>

## 1. 프로젝트 주요 기능

- **대화 처리 파이프라인 (Slot Filling)**: 사용자의 필수 정보(식사, 건강, 기분, 일정, 수면)를 우선적으로 확인하는 체계적인 대화 로직을 수행합니다.
- **의도 분류 (Intent Classification)**: SLM(Small Language Model)을 활용하여 사용자의 의도를 정확히 파악하고 대응합니다.
- **응급 상황 대응**: 위험 징후 포착 시 즉시 상담사 연결 또는 긴급 알림 메시지를 전송합니다.
- **장기 기억 (Long-term Memory)**: FAISS를 활용하여 과거 대화 내용을 기억하고 개인화된 대화 경험을 제공합니다.
- **실시간 음성 스트리밍**: Twilio와 WebSocket을 연동하여 지연 없는 실시간 음성 대화를 지원합니다.
- **비동기 작업 처리**: AWS SQS와 전용 Worker를 통해 부하가 큰 작업을 효율적으로 분산 처리합니다.

<br><br>

## 2. 기술 스택 (Tech Stack)

### Backend & Framework
- **Language**: Python 3.12+
- **Framework**: FastAPI
- **Dependency Injection**: Dependency-injector
- **Async Tasks**: AWS SQS, boto3

### AI & LLM
- **LLM**: OpenAI GPT-4o / GPT-4o-mini
- **Orchestration**: LangChain, LangGraph
- **Memory**: FAISS (Personalized Long-term Memory), Redis (Semantic Cache)
- **STT/TTS**: Clova STT, Luxia TTS
- **Vector DB**: Milvus / Zilliz Cloud (Pymilvus)

### Cache & Database
- **Semantic Cache**: Redis (코사인 유사도 기반 LLM 응답 캐싱)
- **Session Store**: Redis + LangGraph RedisSaver (대화 세션 체크포인트)
- **Connection Pool**: DBUtils PooledDB (MySQL 커넥션 재사용)
- **Chat Logs**: AWS RDS MySQL (`chatbot_logs` 테이블)

### DevOps & Tools
- **Package Manager**: Poetry
- **Container**: Docker, Docker-compose
- **Logging**: Loguru
- **Testing**: Pytest

<br><br>

## 3. 프로젝트 구조 (Folder Structure)

```text
C:\sesac_final\SilverLink-AI
├── app/
│   ├── api/             # API 엔드포인트 핸들러 (Callbot, Chatbot, OCR)
│   ├── callbot/         # 콜봇 도메인 로직 (Service, Repository, Model)
│   ├── chatbot/         # 챗봇 도메인 로직
│   ├── ocr/             # OCR 관련 비즈니스 로직
│   ├── core/            # 프로젝트 핵심 설정 (Config, Container, Middleware)
│   ├── integration/     # 외부 서비스 연동 (LLM, STT, TTS, Call)
│   ├── queue/           # AWS SQS 연동 및 Worker 로직
│   └── util/            # 공통 유틸리티 (Logging, Http Client)
├── tests/               # 유닛 및 통합 테스트
├── docker-compose.yml   # 인프라 구성 (Milvus 등)
├── pyproject.toml       # Poetry 의존성 관리
└── worker_main.py       # SQS 비동기 워커 실행 진입점
```

<br><br>

## 4. 설치 및 실행 방법

### 4.1 사전 요구 사항
- Python 3.12 이상
- Poetry 설치 (`pip install poetry`)
- `.env` 파일 설정 (OpenAI API Key, AWS Credentials, Milvus Host 등)

### 4.2 의존성 설치
```bash
poetry install
```

### 4.3 API 서버 실행
```bash
# Windows 배치 파일 사용 시
run_api.bat

# 또는 직접 실행
python -m uvicorn app.main:app --host 0.0.0.0 --port 5000 --reload
```
- API 문서: `http://localhost:5000/docs`

### 4.4 SQS 워커 실행 (비동기 처리용)
```bash
# Windows 배치 파일 사용 시
run_worker.bat

# 또는 직접 실행
python worker_main.py
```
<br><br>

## 5. 개발 및 테스트 가이드

- **로깅**: `loguru`를 사용하여 로그를 기록하며, `logs/` 디렉토리에 파일로 저장됩니다.
- **테스트 실행**: `pytest`를 사용하여 전체 테스트를 수행할 수 있습니다.
  ```bash
  pytest
  ```
- **코드 스타일**: `ruff`를 사용하여 코드 린팅 및 포맷팅을 관리합니다.

<br><br>



