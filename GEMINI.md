# SilverLink-AI Project Rules & Context

이 문서는 SilverLink-AI 프로젝트의 기술 스택, 아키텍처 및 핵심 개발 규칙을 정의합니다. 모든 코드 작성 및 수정 시 이 규칙을 준수해야 합니다.

## 1. 기술 스택 (Tech Stack)
- **Framework**: FastAPI (Python 3.12+)
- **AI & LLM**: OpenAI API, LangChain, LangGraph, Mem0 (장기 기억)
- **Communication**: Twilio (Websocket 기반 실시간 음성 스트리밍)
- **Database & Vector Search**: Milvus (Vector DB), Pymilvus
- **Messaging Queue**: AWS SQS (boto3)
- **DevOps & Tools**: Docker, Poetry, Pytest, Loguru

## 2. 폴더 구조 규정 (Folder Structure)
- `app/api/endpoints/`: API 엔드포인트 핸들러 (Callbot, Chatbot, OCR)
- `app/callbot/`, `app/chatbot/`, `app/ocr/`: 도메인별 비즈니스 로직 (model, repository, schema, services)
- `app/core/`: 프로젝트 핵심 설정 (config, container, middleware)
- `app/integration/`: 외부 연동 서비스 (LLM, STT, TTS)
- `app/queue/`: AWS SQS 연동 및 비동기 워커 로직
- `app/util/`: 공통 유틸리티 (logging, helpers)

## 3. 핵심 비즈니스 로직 및 개발 규칙

### 3.1 대화 처리 파이프라인 (Strict Sequence)
사용자 입력 처리 시 다음 순서와 조건을 반드시 엄격하게 준수해야 합니다.

1.  **필수 질문 및 슬롯 필링 (Slot Filling)**:
    - **최우선 순위**: 식사 여부, 건강 상태, 기분, 하루 일정, 수면 상태에 대한 필수 질문이 완료되었는지 확인합니다.
    - 슬롯 필드(slot fields)가 채워지지 않았다면, 해당 정보를 묻는 응답을 먼저 수행합니다.
2.  **의도 분류 (Intent Classification)**:
    - 의미 기반 SLM(Small Language Model)을 사용하여 사용자 의도를 분류합니다.
3.  **조건별 대응**:
    - **응급 상황**: "상담사에게 도움을 요청합니다." 메시지를 즉시 전달합니다.
    - **장기 기억 필요**: Mem0에서 검색된 장기 기억 데이터를 포함하여 LLM 답변을 생성합니다.
    - **일반 상황**: LLM을 통해 자연스러운 답변을 생성합니다.

### 3.2 성능 및 안정성 요구사항
- **응답 속도**: 모든 처리 과정은 **ms(밀리초) 단위**로 수행되어야 합니다.
- **비동기 처리**: 작업 부하가 큰 로직은 AWS SQS와 Worker를 통해 비동기로 처리합니다.
- **결함 허용 (Fault Tolerance)**: DLQ(Dead Letter Queue) 및 예외 처리 시스템을 통해 메시지 유실을 방지합니다.

## 4. 개발 가이드라인
- **의존성 관리**: 새로운 패키지 추가 시 `poetry add`를 사용하고 `pyproject.toml`을 업데이트합니다.
- **로깅**: 표준 `print` 대신 `loguru`를 사용하여 컨텍스트 중심의 로그를 남깁니다.
- **테스트**: 모든 핵심 로직 및 API는 `pytest`를 사용하여 유닛/통합 테스트를 작성합니다.
