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

## 6. 챗봇 성능 최적화 (Chatbot Performance Enhancement)

챗봇 서비스는 응답 속도 향상과 OpenAI API 비용 절감을 목표로 **6가지 최적화**를 적용하였습니다.

<br>

### 6.1 챗봇 요청 처리 흐름

```
보호자 앱  ──POST /api/chatbot/chat──►  FastAPI
                                            │
                                   [1] 임베딩 생성
                                   OpenAI text-embedding-3-small
                                            │
                                   [2] Semantic Cache 조회 (Redis db=1)
                                   코사인 유사도 ≥ 0.95 이면 ──► 즉시 반환 ✅
                                            │ MISS
                                   [3] 병렬 벡터 검색
                                   asyncio.gather(
                                     search_faq(embedding),       ← Milvus top-3
                                     search_inquiry(embedding)    ← Milvus top-2
                                   )  → 약 40~50% 시간 절감
                                            │
                                   [4] 동적 컨텍스트 구성
                                   검색 점수 기반 컨텍스트 개수 결정
                                            │
                                   [5] LangGraph + GPT-4o-mini 호출
                                   압축 프롬프트 + trim(max=1500 tokens)
                                            │
                                   [6] Cache 저장 + 백그라운드 로그 저장
                                   asyncio.create_task() → 응답 블로킹 없음
                                            │
                                        응답 반환
```

<br>

### 6.2 6가지 최적화 상세

| # | 최적화 항목 | 적용 방법 | 효과 |
|---|------------|----------|------|
| **①** | **Semantic Cache** | Redis 코사인 유사도(≥ 0.95) 기반 캐싱 | 유사 질문 재질문 시 LLM 호출 완전 우회 |
| **②** | **동적 컨텍스트 크기** | 검색 점수에 따라 컨텍스트 개수 동적 결정 | 토큰 사용량 최소화 (1~5개 자동 조절) |
| **③** | **비동기 로그 저장** | `asyncio.create_task()` 백그라운드 처리 | DB 저장이 응답을 블로킹하지 않음 |
| **④** | **RDS 커넥션 풀** | DBUtils `PooledDB` (max=10, min=2) | TCP 핸드셰이크 제거 (50~250ms → 1~5ms) |
| **⑤** | **시스템 프롬프트 압축** | 프롬프트 토큰 ~220 → ~120 tokens | OpenAI 비용 약 45% 절감 |
| **⑥** | **대화 히스토리 trim 축소** | `trim_messages` 한도 3000 → 1500 tokens | 컨텍스트 윈도우 비용 50% 절감 |

<br>

### 6.3 병렬 벡터 검색 (최적화 ②)

FAQ와 개인 문의(Inquiry) 검색을 `asyncio.gather()`로 **동시에** 실행합니다.

```
직렬 검색 (기존):
  [FAQ 검색: ~200ms] ──► [Inquiry 검색: ~180ms]  = 380ms

병렬 검색 (현재):
  [FAQ 검색: ~200ms]
  [Inquiry 검색: ~180ms]  (동시 실행)             = ~200ms
                                                   → 약 47% 빠름 ⚡
```

```python
# chatbot_service.py
faq_task     = asyncio.create_task(self._search_faq_async(...))
inquiry_task = asyncio.create_task(self._search_inquiry_async(...))
faq_results, inquiry_results = await asyncio.gather(faq_task, inquiry_task)
```

<br>

### 6.4 Semantic Cache 동작 원리 (최적화 ①)

```
질문 수신
    │
임베딩 생성 (1536dim 벡터)
    │
Redis에서 sc:{guardian_id}:* 키 전체 조회
    │
저장된 각 캐시 항목과 코사인 유사도 계산
    │
유사도 ≥ 0.95 ?  ─── YES ──► 캐시 답변 즉시 반환 (LLM 호출 없음)
    │
   NO
    │
LLM 호출 후 답변을 Redis에 저장 (TTL: 24시간)
```

- **캐시 키**: `sc:{guardian_id}:{md5(질문)}`
- **저장 값**: `{ "emb": [...벡터...], "ans": "답변", "src": ["FAQ"] }`
- **TTL**: 86,400초 (24시간)

<br>

### 6.5 동적 컨텍스트 크기 조절 (최적화 ②)

검색 결과의 최고 유사도 점수에 따라 LLM에 전달할 컨텍스트 개수를 자동으로 결정합니다.

| 최고 유사도 점수 | 컨텍스트 개수 | 예상 토큰 수 |
|:---:|:---:|:---:|
| ≥ 0.90 | 1개 | ~120~200 tokens |
| ≥ 0.80 | 2개 | ~240~400 tokens |
| ≥ 0.70 | 3개 | ~360~600 tokens |
| < 0.70 | 5개 | ~600~1000 tokens |

> 유사도가 높을수록 정확한 단일 정보만 전달하여 불필요한 토큰 소모를 줄입니다.

<br>

### 6.6 최적화 관련 환경변수

`.env` 파일에 아래 항목을 추가 설정합니다.

```env
# Redis 연결
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=
REDIS_DB_CHECKPOINT=0   # LangGraph 대화 세션 저장 DB
REDIS_DB_CACHE=1         # Semantic Cache 저장 DB

# Semantic Cache 설정
SEMANTIC_CACHE_THRESHOLD=0.95   # 캐시 히트 임계값 (0.0 ~ 1.0)
SEMANTIC_CACHE_TTL=86400        # 캐시 유효 시간 (초, 기본 24시간)

# RDS (MySQL) 연결 — 챗봇 로그 저장용
RDS_HOST=localhost
RDS_PORT=3306
RDS_USER=root
RDS_PASSWORD=
RDS_DATABASE=silverlink
```

<br>

### 6.7 챗봇 로그 테이블 (`chatbot_logs`)

모든 대화 내역과 성능 지표가 RDS MySQL에 자동 저장됩니다.

| 컬럼 | 설명 |
|------|------|
| `guardian_user_id` | 보호자 ID |
| `elderly_user_id` | 어르신 ID |
| `session_id` | 대화 스레드 ID (`thread_id`) |
| `user_message_text` | 사용자 질문 |
| `bot_response_text` | AI 답변 |
| `source_type` | 답변 출처 (`FAQ` / `INQUIRY`) |
| `embedding_time_ms` | 임베딩 생성 소요 시간 (ms) |
| `search_time_ms` | 벡터 검색 소요 시간 (ms) |
| `llm_time_ms` | LLM 응답 소요 시간 (ms) |
| `response_time_ms` | 전체 응답 소요 시간 (ms) |
| `model_name` | 사용 모델 이름 (예: `gpt-4o-mini`) |
| `retrieval_score` | 검색 최고 유사도 점수 |
| `qa_similarity_score` | 질문-답변 임베딩 유사도 |
| `created_at` | 대화 발생 시각 |

> 이 데이터를 통해 응답 품질 모니터링 및 지속적 성능 개선이 가능합니다.

<br>

### 6.8 병렬 검색 성능 검증

벤치마크 스크립트로 병렬 vs 직렬 검색 성능을 직접 측정할 수 있습니다.

```bash
python benchmark_parallel_vs_sequential.py
```

**예상 결과:**
```
병렬 검색 평균:  ~242ms
직렬 검색 평균:  ~406ms
⚡ 성능 향상:     약 40% 빠름
```
