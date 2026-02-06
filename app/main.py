import logging
import uvicorn
from contextlib import asynccontextmanager  # [추가됨] Lifespan 관리를 위해 필요
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

# from app.api.v1.routes import routers as v1_routers
# from app.api.v2.routes import routers as v2_routers
from app.api.routes import routers
from app.core.config import configs
from app.core.container import Container
from app.util.class_object import singleton
from app.callbot.services.callbot_service import orchestrator_engine

# 로깅 설정 (Loguru 적용 및 중복 로그 방지)
import sys
from loguru import logger as loguru_logger

class InterceptHandler(logging.Handler):
    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = loguru_logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        loguru_logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

# 기존 핸들러 제거 및 InterceptHandler 등록
logging.basicConfig(handlers=[InterceptHandler()], level=logging.INFO, force=True)

# Loguru 설정 (stderr 출력)
loguru_logger.remove()
loguru_logger.add(sys.stderr, level="INFO")

# httpcore와 httpx의 DEBUG 로그 비활성화
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("twilio").setLevel(logging.WARNING) # Twilio API 전체 로그 비활성화
logging.getLogger("twilio.http").setLevel(logging.WARNING) # Twilio API HTTP 로그 비활성화
logging.getLogger("presidio_analyzer").setLevel(logging.ERROR) # Presidio 경고 숨기기

# logger = logging.getLogger(__name__) # Loguru logger는 전역적으로 사용 가능하므로 제거 혹은 호환성 유지
logger = logging.getLogger("app.main") # 호환성을 위해 유지하되, 로그는 InterceptHandler를 통해 Loguru로 전달됨

@singleton
class AppCreator:
    def __init__(self):
        # [변경됨] Container를 먼저 생성 (의존성 확보)
        self.container = Container()
        # self.db = self.container.db()
        # self.db.create_database()

        # [변경됨] FastAPI 앱 생성 시 lifespan 연결
        self.app = FastAPI(
            title=configs.PROJECT_NAME,
            openapi_url="/openapi.json",
            docs_url="/docs",
            version="0.0.1",
            lifespan=self.lifespan,  # 여기에 새로운 수명 주기 핸들러 등록
        )

        # set cors
        if configs.BACKEND_CORS_ORIGINS:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=[str(origin) for origin in configs.BACKEND_CORS_ORIGINS],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

        @self.app.get("/")
        def root():
            return "service is working"
        
        self.app.include_router(routers, prefix=configs.API_STR)

    # [추가됨] 새로운 Lifespan(수명 주기) 핸들러 정의
    # 기존의 @app.on_event("startup") 로직이 이곳으로 이동했습니다.
    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        # --- [Startup] 서버 시작 시 실행될 코드 ---
        greeting = "안녕하세요! 찬주님 실버링크에서 연락드렸습니다. 잘 지내시죠?"
        
        print("⏳ Warming up Models...")
        try:
            # 1. Orchestrator (Qwen) Warm-up
            if hasattr(orchestrator_engine, "local_llm") and orchestrator_engine.local_llm:
                print("✅ Orchestrator (Qwen) Ready.")
            else:
                print("⚠️ Orchestrator (Qwen) not loaded yet.")

            # 2. OpenAI Warm-up (Real Request)
            llm_service = self.container.llm()
            if hasattr(llm_service, "client"):
                print("🚀 Sending dummy request to OpenAI...")
                await llm_service.aclient.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "hi"}],
                    max_tokens=1
                )
                print("✅ OpenAI Connection Established.")
        except Exception as e:
             print(f"⚠️ Model Warm-up Failed: {e}")
             import traceback
             traceback.print_exc()

        print("⏳ Prefetching Greeting & Goodbye TTS...")
        try:
            # self.container를 직접 사용하여 더 안전함
            tts_service = self.container.tts()
            greeting = "안녕하세요! 찬주님 실버링크에서 연락드렸습니다. 잘 지내시죠?"
            bye_msg = "네, 알겠습니다. 어르신, 편히 쉬시고 다음에 또 목소리 들려주세요. 건강하세요!"
            
            await tts_service.asultlux(greeting)
            await tts_service.asultlux(bye_msg)
            print("✅ Greeting & Goodbye TTS Prefetched!")
            
            # [추가] 백엔드 로그인 시도
            callbot_service = self.container.callbot_service()
            await callbot_service._login_backend()
            print("✅ Backend Login successful on startup.")
        except Exception as e:
            print(f"⚠️ Startup pre-processing failed: {e}")
        
        yield  # 서버가 실행되는 동안 여기서 대기합니다.
        
        # --- [Shutdown] 서버 종료 시 실행될 코드 ---
        print("서버를 종료합니다.")

app_creator = AppCreator()
app = app_creator.app
# db = app_creator.db
container = app_creator.container

# 참고: 기존의 @app.on_event("startup") 코드는 모두 위쪽 lifespan 메서드 안으로 통합되어 삭제되었습니다.

print(f'Documents: http://localhost:{configs.PORT}/docs')

if __name__ == '__main__':
    import asyncio
    import platform
    
    # Windows에서 ProactorEventLoop 에러 방지를 위한 설정
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    uvicorn.run("app.main:app", host="0.0.0.0", port=configs.PORT, reload=False, loop="asyncio")