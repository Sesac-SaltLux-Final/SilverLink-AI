import time
import asyncio
import json
import re
import os
import multiprocessing
from typing import List, AsyncGenerator, Dict, Optional, Literal, Any
import urllib.parse
import wave
import io
import traceback
import boto3
import requests
import uuid # UUID 추가
from twilio.rest import Client as TwilioClient
from datetime import datetime
from loguru import logger
from qdrant_client import QdrantClient # Qdrant 직접 제어
from qdrant_client.models import PointStruct

# Disable Mem0 Telemetry to prevent PostHog connection errors
os.environ["MEM0_TELEMETRY"] = "false"

from pydantic import BaseModel, Field
from huggingface_hub import hf_hub_download
from llama_cpp import Llama, LlamaGrammar

# [New] Presidio Imports for PII Filtering
try:
    from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig
    PRESIDIO_AVAILABLE = True
except ImportError:
    print("⚠️ Presidio library not found. PII filtering will be disabled.")
    PRESIDIO_AVAILABLE = False

try:
    from mem0 import Memory
    MEM0_AVAILABLE = True
except ImportError:
    print("⚠️ Mem0 library not found. Memory feature will be disabled.")
    MEM0_AVAILABLE = False

from app.callbot.repository.callbot_repository import CallbotRepository
from app.callbot.services.base_service import BaseService
from app.integration.llm.openai_client import LLM
from app.integration.tts.luxia_client import TTS
from app.integration.call import CALL
from app.core.config import configs
from app.util.http_client import send_data_to_backend

# --- Data Models (From Orchestrator) ---
class SlotItem(BaseModel):
    category: Literal["식사 여부", "건강 상태", "기분", "하루 일정", "수면 상태"] = Field(
        description="""
        정확한 카테고리를 선택하세요:
        - '식사 여부': 밥, 식사, 끼니, 아침/점심/저녁, 배고픔 등 먹는 행위와 관련된 모든 것.
        - '건강 상태': 아픔, 통증, 병원, 약, 컨디션, 몸 상태 등 신체적 건강 관련.
        - '기분': 행복, 슬픔, 외로움, 즐거움 등 감정적 상태. (단, '배불러서 좋다'는 '식사 여부'와 '기분' 둘 다 가능하나, 밥을 먹었다는 사실은 '식사 여부'가 우선임)
        - '하루 일정': 복지관, 산책, 외출, 손님 방문, TV 시청 등 활동 계획.
        - '수면 상태': 잠, 불면증, 꿈, 피곤함 등 수면 관련.
        """
    )
    value: str = Field(description="사용자의 발화 내용을 요약한 값 (예: '밥 먹음', '허리가 아픔')")

class DialogueDecision(BaseModel):
    acknowledgment: str = Field(description="어르신의 말에 대한 공감과 과거 기억을 연결한 문장. (예: '목소리가 밝으셔서 다행이에요. 지난번에 무릎 아프다고 하셔서 걱정했거든요.')")
    question: str = Field(description="다음에 물어볼 질문. (예: '오늘은 좀 어떠세요?')")
    next_action: str = Field(description="'DEEP_DIVE' 또는 'SLOT_QUESTION'")
    topic: Optional[str] = Field(description="현재 주제")

class UnifiedAnalysisResult(BaseModel):
    extracted_slots: List[SlotItem] = Field(description="사용자가 '명시적으로' 언급한 정보만 추출하세요. 추측 금지. 없으면 빈 리스트 [] 반환.")
    # 이제 대답은 Fast LLM이 하므로 여기서는 분석만 함 (Optional 처리)
    dialogue_decision: Optional[DialogueDecision] = Field(None, description="분석용 (생략 가능)")

# --- Global State & Configuration ---
MODEL_NAME = "klue/roberta-small"
EMBEDDING_MODEL_NAME = "jhgan/ko-sroberta-multitask"
MANDATORY_SLOTS = ["식사 여부", "건강 상태", "기분", "하루 일정", "수면 상태"]
MAX_DEEP_DIVE_TURNS = 2

# Global Singleton for Heavy Models (Loaded once)
class OrchestratorEngine:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(OrchestratorEngine, cls).__new__(cls)
            cls._instance.initialize()
        return cls._instance
    
    def initialize(self):
        print("🚀 [OrchestratorEngine] Initializing...")
        self.presidio_analyzer = None
        self.presidio_anonymizer = None
        self.local_llm = None
        self.memory = None
        
        # 1. Initialize Presidio
        if PRESIDIO_AVAILABLE:
            try:
                configuration = {
                    "nlp_engine_name": "spacy",
                    "models": [{"lang_code": "ko", "model_name": "ko_core_news_lg"}],
                }
                provider = NlpEngineProvider(nlp_configuration=configuration)
                nlp_engine = provider.create_engine()
                
                # Presidio 로거 자체를 조용하게 만듦
                # logging.getLogger("presidio_analyzer").setLevel(logging.ERROR)
                
                analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["ko"])
                
                # Patterns
                korean_phone_pattern = Pattern(name="korean_phone_pattern", regex=r"01[016789][-.\]s]?\d{3,4}[-.\]s]?\d{4}", score=1.0)
                analyzer.registry.add_recognizer(PatternRecognizer(supported_entity="PHONE_NUMBER", patterns=[korean_phone_pattern], supported_language="ko"))
                
                korean_rrn_pattern = Pattern(name="korean_rrn_pattern", regex=r"\d{6}[-.\]s]?[1-4]\d{6}", score=1.0)
                analyzer.registry.add_recognizer(PatternRecognizer(supported_entity="KOREAN_RRN", patterns=[korean_rrn_pattern], supported_language="ko"))
                
                self.presidio_analyzer = analyzer
                self.presidio_anonymizer = AnonymizerEngine()
                print("✅ Presidio PII Engine Ready.")
            except Exception as e:
                print(f"⚠️ Presidio Init Failed: {e}")

        # 2. Initialize Local LLM (Qwen)
        print("🚀 [초경량 모드] Qwen 2.5 0.5B Q8_0 로드 중...")
        try:
            model_path = hf_hub_download(
                repo_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
                filename="qwen2.5-0.5b-instruct-q8_0.gguf"
            )
            cores = multiprocessing.cpu_count()
            self.local_llm = Llama(model_path=model_path, n_ctx=1024, n_threads=min(cores, 4), verbose=False)
            print("✅ Local LLM Ready.")
        except Exception as e:
            print(f"⚠️ Local LLM Load Failed: {e}")

        # 3. Initialize Memory
        if MEM0_AVAILABLE:
            try:
                from mem0 import Memory
                
                # 절대 경로 확보
                abs_db_path = os.path.join(configs.PROJECT_ROOT, "mem_db")
                os.makedirs(abs_db_path, exist_ok=True)
                
                print(f"📁 [Mem0] Setting standard path: {abs_db_path}")
                
                # Mem0 표준 설정 방식 (path 문자열 직접 전달)
                mem0_config = {
                    "vector_store": {
                        "provider": "qdrant", 
                        "config": {
                            "path": abs_db_path,
                            "collection_name": "silverlink_memories",
                            "embedding_model_dims": 768, # [Critical] 여기에 설정해야 함
                        }
                    },
                    "llm": {
                        "provider": "openai",
                        "config": {
                            "model": "gpt-4o-mini",
                            "temperature": 0.1
                        }
                    },
                    "embedder": {
                        "provider": "huggingface", 
                        "config": {
                            "model": EMBEDDING_MODEL_NAME,
                        }
                    }
                }
                self.memory = Memory.from_config(mem0_config)
                print(f"✅ Mem0 Memory Initialized with path: {abs_db_path}")
            except Exception as e:
                print(f"⚠️ Memory Load Failed: {e}")
                traceback.print_exc()

# Call Session State Manager (In-Memory for Demo)
# In production, use Redis.
class CallSession:
    _sessions: Dict[str, Dict] = {}

    @classmethod
    def get_session(cls, call_sid: str):
        if call_sid not in cls._sessions:
            cls._sessions[call_sid] = {
                "elderly_id": None,
                "elderly_name": None,
                "call_id": None, # Backend Call ID
                "slots": {slot: None for slot in MANDATORY_SLOTS},
                "deep_dive_count": 0,
                "current_topic": None,
                "last_action": None,
                "history": [],
                "final_analysis": None,
                "analysis_ready": False,
                "recording_ready": False,
                "s3_uri": None,
                "final_duration": 0
            }
        return cls._sessions[call_sid]

    @classmethod
    def update_session(cls, call_sid: str, data: Dict):
        cls._sessions[call_sid] = data
        
    @classmethod
    def clear_session(cls, call_sid: str):
        if call_sid in cls._sessions:
            del cls._sessions[call_sid]

orchestrator_engine = OrchestratorEngine()


class CallbotService(BaseService):
    def __init__(self, callbot_repository: CallbotRepository, llm:LLM, call:CALL, tts:TTS):
        self.callbot_repository= callbot_repository
        self.llm_client = llm
        self.gpt = llm.gpt 
        self.call = call
        self.tts_client = tts
        self.luxia = tts.sultlux
        # [추가] u-law 전용 캐시 (텍스트 -> u-law 바이트)
        self.ulaw_cache = {}
        super().__init__(callbot_repository)
        
    def test(self):
        print('test')
        
    async def _generate_personalized_greeting(self, elderly_name: str, memories: str) -> str:
        """장기 기억을 바탕으로 자연스러운 첫 인사말 생성"""
        name = elderly_name or "어르신"
        prompt = f"""
        역할: 노인 돌봄 AI 상담사 (실버링크)
        상황: 어르신에게 안부 전화를 거는 첫 순간
        어르신 성함: {name}
        과거 기억: {memories}
        
        미션: 과거 기억 속의 '인물'과 '사건'을 포착하여, 아주 반갑고 따뜻한 첫 인사말 한 문장을 작성하세요.
        
        지침:
        1. 반드시 "지난번에"로 시작하여 과거 대화 내용을 자연스럽게 연결하세요.
        2. 기억 내용이 "오늘 딸이 왔어"라면 "지난번에 따님께서 놀러 오셨다고 하셨는데"와 같이 인물(따님)과 사건(놀러 오심) 중심으로 정중하게 변환하세요.
        3. 말투: 해요체 (친절, 따뜻, 공손하게)
        4. 형식: "안녕하세요! {name}님, 반갑습니다. 지난번에 [인물/사건 중심 기억 언급] 하셨는데, 그동안 별일 없으셨어요?"
        5. 길이: 60자 내외의 자연스러운 한 문장.
        """
        try:
            response = await self.llm_client.aclient.chat.completions.create(
                model=configs.INFERENCE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.8
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"⚠️ Greeting Generation Error: {e}")
            return f"안녕하세요! {name}님, 반갑습니다. 잘 지내셨죠?"

    async def build_greeting_gather_twiml(self, call_sid: str, elderly_id: str = None, elderly_name: str = None, phone_number: str = None, initial_mem: str = "", greeting: str = ""):
        # Reset Session
        CallSession.clear_session(call_sid)
        session = CallSession.get_session(call_sid)
        session["elderly_id"] = elderly_id
        session["elderly_name"] = elderly_name

        # [Modified] Check if call_id already exists (created by make_call)
        # If not, create it here (fallback for direct Twilio calls)
        call_id = session.get("call_id")
        
        if not call_id and elderly_id:
            try:
                p_num = phone_number if phone_number else "unknown"
                call_id = await self._send_start_call_to_backend(elderly_id, elderly_name, p_num)
                if call_id:
                    session["call_id"] = call_id
            except Exception as e:
                print(f"❌ [Call Start] Backend Error: {e}")
        elif call_id:
            print(f"ℹ️ [Call Start] Using existing Call ID: {call_id}")
        
        # [Improved] 인사말 결정 로직 (Pre-computed vs On-the-fly)
        final_greeting = ""
        
        # 1. 미리 생성된 인사말이 있으면 우선 사용 (속도 최적화)
        if greeting and len(greeting) > 5:
            final_greeting = greeting
        else:
            # 2. 없으면 기존 로직대로 즉석 생성 (Fallback)
            memories_text = ""
            if orchestrator_engine.memory and elderly_id:
                try:
                    mem_user_id = f"elderly_{elderly_id}"
                    all_mems = orchestrator_engine.memory.get_all(user_id=mem_user_id)
                    res_list = []
                    if isinstance(all_mems, dict) and "results" in all_mems:
                        res_list = all_mems["results"]
                    elif isinstance(all_mems, list):
                        res_list = all_mems
                    
                    if res_list:
                        facts = [m.get('memory', m.get('data', '')) for m in res_list if m]
                        memories_text = ", ".join(facts[:2]) # 핵심 기억 2개만 추출
                except Exception as e:
                    print(f"⚠️ [Memory Retrieval] Error searching memories: {e}")

            if memories_text:
                final_greeting = await self._generate_personalized_greeting(elderly_name, memories_text)
            else:
                name_part = f"{elderly_name} 어르신 " if elderly_name else ""
                final_greeting = f"안녕하세요! {name_part} 반갑습니다. 실버링크에서 연락드렸습니다."

        # [New] Save First Greeting to Backend
        if call_id:
            try:
                await self._send_message_to_backend(call_id, "CALLBOT", greeting)
                print("✅ [Call Start] Saved initial greeting to backend.")

                # # 첫 인사말 저장
                # await self._send_message_to_backend(call_id, "CALLBOT", final_greeting)

            except Exception as e:
                print(f"⚠️ [build_greeting] Failed to save greeting message: {e}")

        encoded_greeting = urllib.parse.quote(final_greeting)
        
        # Initial greeting is pure TTS
        stream_url = f"{configs.CALL_CONTROLL_URL}/api/callbot/stream_response?text={encoded_greeting}&amp;call_sid={call_sid}&amp;mode=tts&amp;elderly_id={elderly_id}"

        # [Updated] TwiML 구조 개선: 명확한 Gather 설정 및 힌트 추가
        # bargeIn=true: 말하면 바로 듣기 시작
        # timeout=5: 말 끝난 후 5초 대기
        twiml = f"""
        <Response>
            <Gather input="speech" action="/api/callbot/gather?elderly_id={elderly_id}" method="POST" language="ko-KR" speechTimeout="auto" bargeIn="true" timeout="5" speechModel="phone_call">
                <Play contentType="audio/basic">{stream_url}</Play>
            </Gather>
            <Redirect>/api/callbot/gather?elderly_id={elderly_id}&amp;retry=0</Redirect>
        </Response>
        """
        return twiml
        
    async def make_call(self, elderly_id: int, phone_number: str, elderly_name: str):
        """[Improved] 전화를 걸기 전, 인삿말을 미리 생성(Pre-compute)하여 지연 시간 제거"""
        print(f"📞 [Make Call] Preparing call for {elderly_name} ({elderly_id})...")
        
        initial_mem = ""
        pre_generated_greeting = ""
        
        # 1. 기억 검색
        try:
            all_mems = self.get_memories(elderly_id)
            res_list = all_mems.get("results", []) if isinstance(all_mems, dict) else all_mems
            
            memories_text = ""
            if res_list:
                facts = [m.get('memory', m.get('data', '')) for m in res_list if m]
                # 최신 기억 전달용
                initial_mem = facts[-1] if facts else ""
                # 인삿말 생성용 (핵심 2개)
                memories_text = ", ".join(facts[:2])
                
            # 2. 인삿말 미리 생성 (여기서 시간 소요)
            print("⏳ [Make Call] Pre-generating greeting...")
            if memories_text:
                pre_generated_greeting = await self._generate_personalized_greeting(elderly_name, memories_text)
            else:
                name_part = f"{elderly_name} 어르신 " if elderly_name else ""
                pre_generated_greeting = f"안녕하세요! {name_part} 반갑습니다. 실버링크에서 연락드렸습니다."
                
            print(f"✅ [Make Call] Greeting ready: {pre_generated_greeting}")
            
            # 3. 음성 미리 생성 (TTS 캐싱)
            print("⏳ [Make Call] Pre-generating audio (TTS Cache)...")
            await self.generate_tts_stream(pre_generated_greeting)
            print("✅ [Make Call] Audio cached successfully.")
            
        except Exception as e:
            print(f"⚠️ [Make Call] Error during pre-computation: {e}")
            pass

        # 4. 통화 연결 (생성된 인삿말 전달)
        # 이제 전화가 연결되면, 서버는 LLM과 TTS API를 모두 기다리지 않고 캐시된 데이터를 즉시 송출합니다.
        return self.call.calling(elderly_id, phone_number, elderly_name, initial_mem, greeting=pre_generated_greeting)

    # --- Backend Communication (Token based) ---

    async def _login_backend(self):
        """로그인하여 Access Token을 발급받고 configs를 업데이트"""
        admin_id = configs.ADMIN_ID
        admin_pw = configs.ADMIN_PW
        
        url = f"{configs.SPRING_BOOT_URL}/api/auth/login"
        payload = {
            "loginId": admin_id,
            "password": admin_pw
        }
        
        print(f"🔑 [Backend Login] Attempting login to {url} with ID: {admin_id}")
        res = await send_data_to_backend(url, payload)
        
        if res and isinstance(res, dict):
            # Handle both direct and nested accessToken
            access_token = res.get("accessToken") or res.get("data", {}).get("accessToken")
            
            if access_token:
                if access_token.startswith("Bearer "):
                    access_token = access_token[7:]
                
                configs.SPRING_BOOT_API_TOKEN = access_token
                print("✅ [Backend Login] Successfully obtained access token.")
                return True
        
        print(f"❌ [Backend Login] Failed to obtain access token. Response: {res}")
        return False

    async def _get_auth_headers(self):
        """인증 헤더 생성. 토큰이 없으면 로그인을 시도."""
        if not configs.SPRING_BOOT_API_TOKEN:
            await self._login_backend()
            
        headers = {"Content-Type": "application/json"}
        if configs.SPRING_BOOT_API_TOKEN:
            token = configs.SPRING_BOOT_API_TOKEN
            # Ensure Bearer prefix is only added once
            if not token.startswith("Bearer "):
                token = f"Bearer {token}"
            headers["Authorization"] = token
        return headers

    async def _call_backend_api(self, url: str, payload: dict, method: str = "POST"):
        """백엔드 API 호출을 담당하며, 401 에러 시 자동으로 재로그인 후 1회 재시도합니다."""
        headers = await self._get_auth_headers()
        res = await send_data_to_backend(url, payload, method=method, headers=headers)
        
        # 401 Unauthorized Error Handling (Token expired or invalid)
        if res and isinstance(res, dict) and res.get("error") == "UNAUTHORIZED_401":
            print("⚠️ [Backend API] Auth failed (401). Attempting re-login...")
            configs.SPRING_BOOT_API_TOKEN = None # Clear invalid token
            success = await self._login_backend() # Re-login
            
            if success:
                headers = await self._get_auth_headers() # Get new headers with fresh token
                print(f"🔄 [Backend API] Retrying request to {url}...")
                res = await send_data_to_backend(url, payload, method=method, headers=headers)
            else:
                print("❌ [Backend API] Re-login failed. Cannot retry request.")
        
        return res

    async def _send_start_call_to_backend(self, elderly_id, name, phone_number):
        """통화 시작 API 호출 -> call_id 반환"""
        url = f"{configs.SPRING_BOOT_URL}/api/internal/callbot/calls"
        payload = {
            "elderlyId": int(elderly_id),
            "name": name,
            "phoneNumber": phone_number,
            "callAt": datetime.now().isoformat()
        }
        res = await self._call_backend_api(url, payload)
        
        if res and "data" in res and "callId" in res["data"]:
            return res["data"]["callId"]
        return None

    async def _send_message_to_backend(self, call_id: int, speaker: str, content: str, danger: bool = False, danger_reason: str = None):
        """대화 메시지 저장 API 호출 (CALLBOT 또는 ELDERLY)"""
        if not call_id: 
            return
        url = f"{configs.SPRING_BOOT_URL}/api/internal/callbot/calls/{call_id}/messages"
        payload = {
            "speaker": speaker,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "danger": danger,
            "dangerReason": danger_reason
        }
        await self._call_backend_api(url, payload)

    async def _send_end_call_to_backend(self, call_id: int, duration: int, summary: str, emotion: str, daily_status: dict, recording_url: str = None):
        """통화 종료 API 호출 (EndCallRequest DTO 일치)"""
        if not call_id: 
            return
        url = f"{configs.SPRING_BOOT_URL}/api/internal/callbot/calls/{call_id}/end"
        
        # 제공해주신 EndCallRequest DTO 구조와 100% 일치시킴
        payload = {
            "callTimeSec": int(duration),   
            "recordingUrl": recording_url,
            "summary": {"content": summary},
            "emotion": {"emotionLevel": emotion},
            "dailyStatus": daily_status
        }
        
        await self._call_backend_api(url, payload)

    # --- Orchestrator Logic ---
    async def anonymize_text_async(self, text: str) -> str:
        def _run():
            processed_text = text
            if orchestrator_engine.presidio_analyzer and orchestrator_engine.presidio_anonymizer:
                try:
                    results = orchestrator_engine.presidio_analyzer.analyze(
                        text=processed_text,
                        entities=["PHONE_NUMBER", "KOREAN_RRN", "EMAIL_ADDRESS", "PERSON"],
                        language='ko' 
                    )
                    anonymized_result = orchestrator_engine.presidio_anonymizer.anonymize(
                        text=processed_text,
                        analyzer_results=results,
                        operators={
                            "DEFAULT": OperatorConfig("replace", {"new_value": "<REDACTED>"}),
                            "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "<전화번호>"}),
                            "KOREAN_RRN": OperatorConfig("replace", {"new_value": "<주민번호>"}),
                            "PERSON": OperatorConfig("replace", {"new_value": "<이름>"}),
                        }
                    )
                    processed_text = anonymized_result.text
                except Exception as e:
                    print(f"⚠️ Presidio PII Error: {e}")

            phone_regex = r"01[016789][-.\]s]?\d{3,4}[-.\]s]?\d{4}|0\d{1,2}[-.\]s]?\d{3,4}[-.\]s]?\d{4}"
            processed_text = re.sub(phone_regex, "<전화번호>", processed_text)
            rrn_regex = r"\d{6}[-.\]s]?[1-4]\d{6}"
            processed_text = re.sub(rrn_regex, "<주민번호>", processed_text)
            return processed_text
        return await asyncio.to_thread(_run)

    async def get_intent_async(self, text: str) -> str:
        clean_text = text.strip()
        if len(clean_text) <= 5: 
            return "GENERAL"
        
        emergency_keywords = ["살려줘", "숨이 안", "숨 못", "가슴이 아파", "쓰러졌", "119", "죽을 것 같", "도와줘", "큰일났어"]
        if any(k in clean_text for k in emergency_keywords):
            return "EMERGENCY"

        if orchestrator_engine.local_llm:
            gbnf_grammar = r'root ::= ("General" | "Emergency")'
            grammar = LlamaGrammar.from_string(gbnf_grammar)
            prompt = f"""<|im_start|>system
You are a text classifier. Classify: 'General' or 'Emergency'.
<|im_end|>
<|im_start|>user
Input: "{clean_text}"
Output:<|im_end|>
<|im_start|>assistant
"""
            def _run_llm():
                try:
                    return orchestrator_engine.local_llm(
                        prompt, max_tokens=5, temperature=0.0, stop=["<|im_end|>"], grammar=grammar
                    )
                except Exception as e:
                    print(f"LLM Error: {e}")
                    return None
            output = await asyncio.to_thread(_run_llm)
            if output:
                return output["choices"][0]["text"].strip()
        return "GENERAL"

    async def process_conversation(self, call_sid: str, elderly_id: str, raw_user_input: str) -> Dict[str, Any]:
        """
        Main Orchestrator Logic: PII -> Intent -> Unified LLM -> Memory
        Returns: { "intent": str, "response": str, "session": dict }
        """
        from app.util.log import log_detailed
        start_total = time.time()
        session = CallSession.get_session(call_sid)
        timeouts = {}
        
        # [Updated] call_id를 최상단에서 정의하여 모든 경로에서 사용 가능하게 함
        call_id = session.get("call_id")

        # [최적화] 종료 키워드 즉시 감지 (가장 먼저 수행)
        exit_keywords = ["그만", "그만해", "됐어", "종료", "끊어", "끊을게", "다음에하자", "또전화", "다음에연락"]
        clean_input = raw_user_input.replace(" ", "")
        
        # 종료 감지 로직 (예외 처리 추가: '끊어졌어' 등)
        is_exit = any(kw in clean_input for kw in exit_keywords)
        if "끊어졌어" in clean_input:
            is_exit = False
            
        if is_exit:
            print(f"🛑 [Fast Exit] Termination detected immediately: {raw_user_input}")
            final_response = "네, 알겠습니다. 어르신, 편히 쉬시고 다음에 또 목소리 들려주세요. 건강하세요!"
            
            if "history" not in session: session["history"] = []
            session["history"].append({"user": raw_user_input, "ai": final_response})
            
            if call_id:
                # 1. User message (Immediate & Await)
                await self._send_message_to_backend(call_id, "ELDERLY", raw_user_input)
                
                # 2. Bot message (Delayed 1s & Await)
                await asyncio.sleep(1) # Force 1s delay
                await self._send_message_to_backend(call_id, "CALLBOT", final_response)
            
            CallSession.update_session(call_sid, session)
            asyncio.create_task(self.finalize_call(call_sid, "0"))
            
            return {"intent": "END_CALL", "response": final_response, "session": session}

        # [Updated] 대화 턴 수 계산 (현재 턴 포함)
        current_turn_count = len(session.get("history", [])) + 1

        # 1. PII Filtering & Memory Retrieval (정상 경로)
        # Fast Exit을 통과한 경우에만 수행
        timeouts['stt_processing'] = 'External (Twilio)'
        t_pii_start = time.time()
        user_input = await self.anonymize_text_async(raw_user_input)
        timeouts['pii_filtering'] = time.time() - t_pii_start

        # Memory Retrieval
        t_mem_search_start = time.time()
        relevant_memories_text = "No relevant memories."
        if orchestrator_engine.memory:
            try:
                str_eid = str(elderly_id)
                mem_user_id = f"elderly_{str_eid}"
                all_mems = orchestrator_engine.memory.get_all(user_id=mem_user_id)
                
                results_list = []
                if isinstance(all_mems, dict) and "results" in all_mems:
                    results_list = all_mems["results"]
                elif isinstance(all_mems, list):
                    results_list = all_mems
                
                if results_list:
                    facts = [m.get('memory', m.get('data', '')) for m in results_list if m]
                    relevant_memories_text = "\n".join([f"- {f}" for f in facts if f])
            except Exception:
                pass
        timeouts['memory_search'] = time.time() - t_mem_search_start

        # 2. Intent Classification
        t_intent_start = time.time()
        intent = await self.get_intent_async(user_input)
        timeouts['intent_check'] = time.time() - t_intent_start
        
        # [Updated] Send User Message to Backend
        if call_id:
            asyncio.create_task(self._send_message_to_backend(call_id, "ELDERLY", raw_user_input, danger=(intent=="EMERGENCY")))
        
        if intent == "EMERGENCY":
            final_response = "어르신 확인했습니다. 안전을 위해 담당 상담사님과 보호자님께 긴급알림을 즉시 전송하겠습니다."
            
            # Update History for Emergency
            if "history" not in session: 
                session["history"] = []
            session["history"].append({"user": user_input, "ai": final_response})
            
            timeouts['total_processing'] = time.time() - start_total
            log_detailed(raw_user_input, user_input, final_response, "EMERGENCY_STOP", "None", 0, {}, timeouts, call_sid)
            return {
                "intent": "EMERGENCY",
                "response": final_response,
                "session": session
            }
            
        # 3. Unified LLM (Logic & Generation)
        t_llm_start = time.time()
        # Calculate Logic State
        current_missing = [s for s, v in session["slots"].items() if v is None]
        target_slot = current_missing[0] if current_missing else "작별 인사 및 건강 당부"
        
        exit_keywords = ["그만", "그만해", "됐어", "종료", "끊어", "끊을게", "다음에하자", "또전화", "다음에연락"]
        clean_input = user_input.replace(" ", "")
        
        # 종료 감지 로직 (예외 처리 추가)
        is_exit_input = any(k in clean_input for k in exit_keywords)
        if "끊어졌어" in clean_input:
            is_exit_input = False
            
        force_slot_question = is_exit_input or (session["deep_dive_count"] >= MAX_DEEP_DIVE_TURNS)
        
        if not current_missing:
            force_slot_question = False
            target_slot = "작별 인사 및 건강 당부"

        unified_system_prompt = f"""
    # MISSION
    You are an Analyst AI. Your ONLY goal is to extract key information (Slots) from the user's input.
    DO NOT generate a response. The response has already been handled by another system.
    
    [Current Status]
    - Turn: {current_turn_count}
    - Missing Slots: {current_missing}
    
    # STEP 1: FACT EXTRACTION (Information Extraction)
    - Scan user input for keywords related to: {MANDATORY_SLOTS}.
    - **Mapping Rules**:
        - "밥 먹었어", "배불러", "입맛 없어" -> [식사 여부]
        - "아파", "쑤셔", "약 먹었어", "병원" -> [건강 상태]
        - "좋아", "슬퍼", "우울해", "심심해" -> [기분]
        - "잤어", "못 잤어", "꿈꿨어" -> [수면 상태]
        - "노인정 갔어", "산책 할거야", "집에 있었어" -> [하루 일정]
    - **Constraint**: Extract ONLY what is explicitly stated. Do NOT guess.
    """
        
        try:
            # Construct Messages with History
            messages = [{"role": "system", "content": unified_system_prompt}]
            recent_history = session.get("history", [])[-4:]
            for turn in recent_history:
                messages.append({"role": "user", "content": turn["user"]})
                messages.append({"role": "assistant", "content": turn["ai"]})
            messages.append({"role": "user", "content": user_input})

            completion = await self.llm_client.aclient.beta.chat.completions.parse(
                model=configs.INFERENCE_MODEL,
                messages=messages,
                response_format=UnifiedAnalysisResult,
                temperature=1.0,
                max_tokens=300, 
            )
            result = completion.choices[0].message.parsed
            timeouts['unified_llm_processing'] = time.time() - t_llm_start
            
            # [Updated] Fast LLM이 생성했던 대답을 가져옴 (없으면 기본값)
            final_response = session.get("last_ai_response", "죄송합니다, 잠시 문제가 생겼어요.")
            print(f"🐢 [Slow Analysis] Using Fast LLM Response: {final_response}")

            # [Updated] Send Bot Message to Backend
            if call_id:
                asyncio.create_task(self._send_message_to_backend(call_id, "CALLBOT", final_response))    
            
            # Update Slots
            any_slot_filled = False
            for item in result.extracted_slots:
                if item.value and str(item.value).lower() not in ["null", "none", "없음"]:
                    session["slots"][item.category] = item.value
                    any_slot_filled = True

            # [Improved] 딥다이브 카운트 로직 개선
            # 1. 현재 목표였던 슬롯(target_slot)이 이번 턴에 채워졌는지 확인
            target_filled = (target_slot in session["slots"] and session["slots"][target_slot] is not None)
            
            if target_filled:
                # 목표 달성! 다음 주제로 넘어가기 위해 카운트 리셋 (또는 딥다이브 종료)
                # 단, 사용자가 너무 짧게 대답했다면 한 번 더 물어볼 수도 있음(선택 사항).
                # 여기서는 깔끔하게 다음으로 넘어가도록 0으로 리셋.
                session["deep_dive_count"] = 0
            elif any_slot_filled:
                # 목표는 아니지만 다른 정보를 줬다면 대화 이어가기 (카운트 증가)
                session["deep_dive_count"] += 1
            else:
                # 아무 정보도 없으면 리셋 (화제 전환 유도)
                session["deep_dive_count"] = 0
            
            # [Safety] 카운트가 너무 커지면 강제 리셋
            if session["deep_dive_count"] > MAX_DEEP_DIVE_TURNS:
                session["deep_dive_count"] = 0
            
            # 다음 타겟 슬롯 계산 (로그용)
            next_missing = [s for s, v in session["slots"].items() if v is None]
            next_target_slot = next_missing[0] if next_missing else "작별 인사 및 건강 당부"

            # Update History
            if "history" not in session:
                session["history"] = []
            session["history"].append({"user": user_input, "ai": final_response})
            
            timeouts['total_processing'] = time.time() - start_total
            
            filled_slots_dict = {s: v for s, v in session["slots"].items() if v is not None}
            log_detailed(
                raw_user_input, user_input, final_response, 
                target_slot, next_target_slot, session["deep_dive_count"], filled_slots_dict, 
                timeouts, call_sid
            )
            
            CallSession.update_session(call_sid, session)
            
            return {
                "intent": "GENERAL",
                "response": final_response, # 이제 여기서는 아무 의미 없지만 형식상 반환
                "session": session
            }

        except Exception as e:
            timeouts['unified_llm_processing'] = time.time() - t_llm_start
            timeouts['total_processing'] = time.time() - start_total
            print(f"Unified LLM Error: {e}")
            traceback.print_exc()
            return {
                "intent": "GENERAL",
                "response": "죄송해요, 잠시 제 귀가 어두웠나봐요. 다시 말씀해 주시겠어요?",
                "session": session
            }

    async def _summarize_conversation(self, history: List[Dict]) -> str:
        """Generates a concise summary of the conversation using OpenAI."""
        if not history: 
            return "대화 내용 없음."
        
        conversation_text = "\n".join([f"User: {turn['user']}\nAI: {turn['ai']}" for turn in history])
        
        prompt = f"""
        Summarize the following conversation with an elderly person in Korean.
        Focus on key information: Meal status, Health condition, Mood, and Schedule.
        Keep it concise (1-2 sentences).
        
        [Conversation]
        {conversation_text}
        """
        
        try:
            response = await self.llm_client.aclient.chat.completions.create(
                model=configs.INFERENCE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Summary Generation Error: {e}")
            return "요약 실패"

    async def _analyze_sentiment_with_llm(self, text: str) -> Optional[str]:
        """Analyzes sentiment (GOOD, BAD, NORMAL) using LLM."""
        if not text: return None
        
        prompt = f"""
        Analyze the sentiment of the following text regarding health or sleep condition.
        Classify into one of three categories: "GOOD", "BAD", "NORMAL".
        
        Text: "{text}"
        
        Output only the category name.
        """
        try:
            response = await self.llm_client.aclient.chat.completions.create(
                model=configs.INFERENCE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.0
            )
            result = response.choices[0].message.content.strip().upper()
            if "GOOD" in result: return "GOOD"
            if "BAD" in result: return "BAD"
            if "NORMAL" in result: return "NORMAL"
            return None
        except Exception as e:
            print(f"Sentiment Analysis Error: {e}")
            return None

    async def _analyze_meal_status_with_llm(self, text: str) -> Optional[bool]:
        """Analyzes meal status (True/False) using LLM."""
        if not text: return None
        
        prompt = f"""
        Determine if the user has eaten a meal based on the text.
        Text: "{text}"
        
        If they ate (or are full), output "TRUE".
        If they did not eat (or skipped), output "FALSE".
        If it's unclear or not mentioned, output "UNKNOWN".
        """
        try:
            response = await self.llm_client.aclient.chat.completions.create(
                model=configs.INFERENCE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5,
                temperature=0.0
            )
            result = response.choices[0].message.content.strip().upper()
            if "TRUE" in result: return True
            if "FALSE" in result: return False
            return None
        except Exception as e:
            print(f"Meal Analysis Error: {e}")
            return None

    async def _map_slots_to_daily_status(self, slots: Dict) -> Dict:
        """Maps slots to DailyStatusRequest format (Async with LLM)"""
        # 1. Meal Analysis with LLM
        meal_text = slots.get("식사 여부")
        meal_taken = await self._analyze_meal_status_with_llm(meal_text)
        
        # 2. Status Analysis with LLM
        health_text = slots.get("건강 상태")
        sleep_text = slots.get("수면 상태")
        
        health_status = await self._analyze_sentiment_with_llm(health_text)
        sleep_status = await self._analyze_sentiment_with_llm(sleep_text)
        
        return {
            "mealTaken": meal_taken,
            "healthStatus": health_status,
            "healthDetail": health_text or "",
            "sleepStatus": sleep_status,
            "sleepDetail": sleep_text or ""
        }

    async def _analyze_overall_emotion(self, history: List[Dict]) -> str:
        """Infers overall emotion from conversation history using LLM"""
        if not history: 
            return None
        
        conversation_text = "\n".join([f"User: {turn['user']}\nAI: {turn['ai']}" for turn in history])
        
        prompt = f"""
        Analyze the overall emotional state of the elderly user in this conversation.
        Classify into one of: "GOOD", "NORMAL", "BAD", "DEPRESSED".
        
        [Conversation]
        {conversation_text}
        
        Output only the category name.
        """
        
        try:
            response = await self.llm_client.aclient.chat.completions.create(
                model=configs.INFERENCE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.0
            )
            result = response.choices[0].message.content.strip().upper()
            valid_emotions = ["GOOD", "NORMAL", "BAD", "DEPRESSED"]
            for emotion in valid_emotions:
                if emotion in result:
                    return emotion
            return "NORMAL"
        except Exception as e:
            print(f"Emotion Analysis Error: {e}")
            return "NORMAL"

    async def _upload_recordings(self, call_sid: str) -> Optional[tuple[str, int]]:
        """Twilio API를 통해 녹음 파일을 찾아 S3에 업로드하고 (URL, 시간) 반환"""
        def _sync_upload():
            try:
                twilio_client = TwilioClient(configs.TWILIO_SID, configs.TWILIO_TOKEN)
                
                # 녹음 파일이 생성될 때까지 최대 10초간 대기 (2초 간격 5회)
                recordings = None
                for i in range(5):
                    recordings = twilio_client.recordings.list(call_sid=call_sid, limit=1)
                    if recordings:
                        break
                    print(f"⏳ [Recording] Waiting for Twilio to process recording... ({i+1}/5)")
                    time.sleep(2)
                
                if not recordings:
                    print(f"⚠️ [Recording] No recordings found for {call_sid}")
                    return None, 0

                record = recordings[0]
                recording_sid = record.sid
                duration = getattr(record, 'duration', 0)
                media_url = f"https://api.twilio.com/2010-04-01/Accounts/{configs.TWILIO_SID}/Recordings/{recording_sid}.mp3"
                
                print(f"📥 [Recording] Downloading {media_url}...")
                response = requests.get(media_url, auth=(configs.TWILIO_SID, configs.TWILIO_TOKEN))
                
                if response.status_code == 200:
                    s3_client = boto3.client(
                        "s3",
                        region_name=configs.AWS_REGION,
                        aws_access_key_id=configs.AWS_ACCESS_KEY_ID,
                        aws_secret_access_key=configs.AWS_SECRET_ACCESS_KEY
                    )
                    file_key = f"private/voice/{recording_sid}.mp3"
                    s3_client.put_object(
                        Bucket=configs.AWS_S3_BUCKET_NAME,
                        Key=file_key,
                        Body=response.content,
                        ContentType="audio/mpeg"
                    )
                    s3_uri = f"s3://{configs.AWS_S3_BUCKET_NAME}/{file_key}"
                    print(f"✅ [Recording] S3 Upload Success: {s3_uri}")
                    return s3_uri, int(duration)
                return None, 0
            except Exception as e:
                print(f"❌ [Recording] Error: {e}")
                return None, 0
        
        return await asyncio.to_thread(_sync_upload)

    async def _perform_final_backend_update(self, call_sid: str, session: dict):
        """세션에 모인 모든 데이터(분석+녹음)를 백엔드에 최종 전송"""
        call_id = session.get("call_id")
        final_data = session.get("final_analysis")
        s3_uri = session.get("s3_uri")
        duration = session.get("final_duration", 0)

        if not call_id or not final_data:
            return

        await self._send_end_call_to_backend(
            call_id, 
            int(duration), 
            final_data["summary"], 
            final_data["emotion"], 
            final_data["daily_status"], 
            s3_uri
        )
        
        # 전송 완료 후 세션 삭제
        CallSession.clear_session(call_sid)

    async def upload_recording_from_url(self, recording_url: str, recording_sid: str, call_sid: str = None, duration: int = None) -> Optional[str]:
        """Twilio Callback에서 받은 URL로 S3에 업로드 후, 분석 결과가 있다면 백엔드 전송"""
        def _sync_upload():
            try:
                media_url = f"{recording_url}.mp3"
                response = requests.get(media_url) 
                if response.status_code == 200:
                    s3_client = boto3.client("s3", region_name=configs.AWS_REGION, 
                                           aws_access_key_id=configs.AWS_ACCESS_KEY_ID, 
                                           aws_secret_access_key=configs.AWS_SECRET_ACCESS_KEY)
                    file_key = f"private/voice/{recording_sid}.mp3"
                    s3_client.put_object(Bucket=configs.AWS_S3_BUCKET_NAME, Key=file_key, 
                                       Body=response.content, ContentType="audio/mpeg")
                    return f"s3://{configs.AWS_S3_BUCKET_NAME}/{file_key}"
                return None
            except Exception as e:
                return None

        s3_uri = await asyncio.to_thread(_sync_upload)
        
        if call_sid:
            session = CallSession.get_session(call_sid)
            session["s3_uri"] = s3_uri
            
            # 콜백으로 온 duration이 있다면 이를 최우선으로 사용 (0보다 클 때만)
            if duration is not None and int(duration) > 0:
                session["final_duration"] = int(duration)
            
            session["recording_ready"] = True
            
            # finalize_call이 이미 분석을 마쳤다면 최종 전송
            if session.get("analysis_ready"):
                await self._perform_final_backend_update(call_sid, session)
        
        return s3_uri

    async def finalize_call(self, call_sid: str, duration: str = "0"):
        """통화 종료 시 분석 수행 후, 녹음이 준비되었다면 백엔드 전송"""
        session = CallSession.get_session(call_sid)
        history = session.get("history", [])
        call_id = session.get("call_id")
        elderly_id = session.get("elderly_id")
        
        if not call_id:
            CallSession.clear_session(call_sid)
            return

        # 1. 대화 분석 (요약, 감정, 상태)
        if history:
            summary = await self._summarize_conversation(history)
            daily_status = await self._map_slots_to_daily_status(session.get("slots", {}))
            emotion = await self._analyze_overall_emotion(history)
            
            # [Updated] Long-term Memory 저장 (어르신 고유 ID 사용)
            if orchestrator_engine.memory:
                mem_user_id = f"elderly_{elderly_id}"
                await self._save_full_history_async(mem_user_id, history)
        else:
            summary = "통화 내용 없음 (짧은 통화)"
            # 값이 없으면 None으로 전송 (백엔드에서 null 처리)
            daily_status = {
                "mealTaken": None, 
                "healthStatus": None, 
                "healthDetail": None, 
                "sleepStatus": None, 
                "sleepDetail": None
            }
            emotion = "NORMAL" # 감정은 분석 불가 시 NORMAL 유지 (또는 None)

        # 2. 분석 결과 세션에 저장
        session["final_analysis"] = {
            "summary": summary,
            "daily_status": daily_status,
            "emotion": emotion
        }
        session["analysis_ready"] = True
        
        # 콜백에서 아직 정확한 시간이 안 왔거나 현재 저장된 시간이 0인 경우에만 업데이트
        new_duration = 0
        try: 
            new_duration = int(duration)
        except Exception: 
            pass
        
        if (session.get("final_duration") or 0) == 0 and new_duration > 0:
            session["final_duration"] = new_duration

        # 3. 녹음 콜백이 이미 도착했는지 확인 후 전송
        if session.get("recording_ready"):
            await self._perform_final_backend_update(call_sid, session)
        else:
            # 안전장치: 만약 30초 내에 녹음 콜백이 안 오면 분석 결과만이라도 전송하도록 예약 가능 (생략 가능)
            async def _safety_fallback():
                await asyncio.sleep(30)
                active_session = CallSession.get_session(call_sid)
                if active_session.get("analysis_ready") and not active_session.get("recording_ready"):
                    await self._perform_final_backend_update(call_sid, active_session)
            asyncio.create_task(_safety_fallback())

    def get_memories(self, elderly_id: int) -> List[Dict]:
        """특정 어르신의 모든 기억 조회"""
        if not orchestrator_engine.memory:
            return []
        
        try:
            user_id = f"elderly_{elderly_id}"
            return orchestrator_engine.memory.get_all(user_id=user_id)
        except Exception as e:
            return []

    async def _save_full_history_async(self, user_id: str, history: List[Dict]):
        """Helper to save summarized facts to Mem0 for better update performance"""
        if not orchestrator_engine.memory: 
            return
            
        def _batch_save():
            try:
                # 1. 이번 대화의 내용을 텍스트로 병합
                conversation_text = ""
                for turn in history:
                    conversation_text += f"사용자: {turn['user']}\n상담사: {turn['ai']}\n"
                
                # 2. Mem0 add 호출 (timestamp를 제거하여 동일 주제 업데이트 유도)
                orchestrator_engine.memory.add(
                    conversation_text, 
                    user_id=user_id,
                    metadata={"source": "callbot"} # 고정된 메타데이터 사용
                )

            except Exception as e:
                pass
        
        await asyncio.to_thread(_batch_save)
    # --- Audio Utils ---
    def wav_to_ulaw(self, wav_bytes: bytes) -> bytes:
        """Converts WAV bytes to raw Mu-law audio (8kHz, Mono) without headers"""
        try:
            import audioop
        except ImportError:
            print("Audioop module is not available. Audio conversion disabled.")
            return b""

        try:
            with wave.open(io.BytesIO(wav_bytes), 'rb') as wav_file:
                n_channels = wav_file.getnchannels()
                framerate = wav_file.getframerate()
                sampwidth = wav_file.getsampwidth()
                n_frames = wav_file.getnframes()
                data = wav_file.readframes(n_frames)

                if n_channels > 1:
                    data = audioop.tomono(data, sampwidth, 0.5, 0.5)
                
                if framerate != 8000:
                    data, _ = audioop.ratecv(data, sampwidth, 1, framerate, 8000, None)

                return audioop.lin2ulaw(data, sampwidth)
        except Exception as e:
            print(f"Audio Conv Error: {e}")
            return b""

    async def generate_tts_stream(self, text: str):
        content = await self.tts_client.asultlux(text)
        return content

    # --- Streaming Logic (Real-time LLM + TTS) ---
    async def ai_response_generator(self, user_input: str, history: List[dict], mode: str = "chat", start_ts: float = 0.0, elderly_id: str = None, call_sid: str = None) -> AsyncGenerator[bytes, None]:
        """
        [Ultra-Fast] LLM Stream -> Text Buffer -> TTS -> Audio Stream
        """
        if not user_input and mode == "chat": 
            return

        try:
            # [Fast Path] u-law 캐시에 있으면 즉시 반환 (지연 최소화)
            if user_input in self.ulaw_cache:
                print(f"⚡ [Fast Path] Serving cached u-law audio for: {user_input[:15]}...")
                yield self.ulaw_cache[user_input]
                return

            # 1. TTS 모드 (단순 텍스트 재생)
            if mode == "tts":
                # 기존 로직 유지 (안내 멘트 등)
                sentences = re.split(r'(?<=[.!?,;])\s+', user_input)
                sentences = [s.strip() for s in sentences if s.strip()]
                
                full_audio = b""
                for i, sentence in enumerate(sentences):
                    wav_data = await self.generate_tts_stream(sentence)
                    if wav_data:
                        ulaw_data = self.wav_to_ulaw(wav_data)
                        full_audio += ulaw_data
                        yield ulaw_data
                
                # 생성된 전체 오디오를 다음에 쓸 수 있도록 캐싱
                if len(user_input) < 200: # 너무 긴 대화는 메모리 절약을 위해 제외
                    self.ulaw_cache[user_input] = full_audio
                return

            # 2. Chat 모드 (실시간 생성)
            print(f"🚀 [Real-time] Generating response for: {user_input}")
            
            # [Added] Emergency Check for Fast LLM
            emergency_keywords = ["살려줘", "숨이 안", "숨 못", "가슴이 아파", "쓰러졌", "119", "죽을 것 같", "도와줘", "큰일났어"]
            if any(k in user_input for k in emergency_keywords):
                emergency_response = "어르신 확인했습니다. 안전을 위해 담당 상담사님과 보호자님께 긴급알림을 즉시 전송하겠습니다."
                wav_data = await self.generate_tts_stream(emergency_response)
                if wav_data:
                    yield self.wav_to_ulaw(wav_data)
                
                if call_sid:
                    session = CallSession.get_session(call_sid)
                    session["last_ai_response"] = emergency_response
                return

            # 장기 기억 검색 (빠르게)
            memory_context = ""
            if orchestrator_engine.memory and elderly_id:
                try:
                    all_mems = orchestrator_engine.memory.get_all(user_id=f"elderly_{elderly_id}")
                    if isinstance(all_mems, dict): all_mems = all_mems.get("results", [])
                    facts = [m.get('memory', '') for m in all_mems if m]
                    memory_context = "\n".join(facts[-3:]) # 최신 3개만
                except: pass

            # [Improved] Fast LLM을 위한 정교한 프롬프트
            session = CallSession.get_session(call_sid)
            filled_slots = [k for k, v in session.get("slots", {}).items() if v is not None]
            missing_slots = [k for k, v in session.get("slots", {}).items() if v is None]
            current_target = missing_slots[0] if missing_slots else "건강 당부"

            system_prompt = f"""
            Role: 노인 돌봄 AI 상담사 (실버링크).
            User Memory: {memory_context}
            Already Known Info: {filled_slots} (Do NOT ask about these again)
            Current Target Topic: {current_target}
            
            # Guidelines (Strict):
            1. **Check First**: If the user's latest input ALREADY answers the 'Current Target Topic' or any 'Missing Topics', do NOT ask about it. Move to a natural reaction instead.
            2. **No Repetition**: Do NOT ask for information that was just provided or is already in 'Already Known Info'.
            3. **Acknowledge and Flow**: [Warm Reaction to what user said] -> [Light Follow-up or move to NEXT topic].
            4. **Natural Transition**: If '식사 여부' is done, naturally move to '건강 상태' or '기분'.
            5. **Tone**: Warm, Respectful, Polite (Haeyo-che).
            6. **Format**: Plain text only. Max 2 sentences.
            """
            
            messages = [{"role": "system", "content": system_prompt}]
            # 최근 대화 2턴만 추가 (Context 줄이기)
            for turn in history[-2:]:
                messages.append({"role": "user", "content": turn['user']})
                messages.append({"role": "assistant", "content": turn['ai']})
            messages.append({"role": "user", "content": user_input})

            # LLM 스트리밍 요청
            stream = await self.llm_client.aclient.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                stream=True,
                max_tokens=150,
                temperature=0.7
            )

            buffer = ""
            full_fast_response = "" # 전체 응답 수집용
            async for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    buffer += content
                    full_fast_response += content
                    # 문장 부호가 나오면 즉시 TTS 요청
                    if any(punct in content for punct in ".!?,;\n"):
                        if len(buffer.strip()) > 2:
                            wav_data = await self.generate_tts_stream(buffer)
                            if wav_data:
                                yield self.wav_to_ulaw(wav_data)
                            buffer = ""
            
            # 남은 버퍼 처리
            if buffer.strip():
                wav_data = await self.generate_tts_stream(buffer)
                if wav_data:
                    yield self.wav_to_ulaw(wav_data)
            
            print(f"\n⚡ [Fast LLM] (User Heard): {full_fast_response}")
            
            # [Crucial] 사용자가 들은 이 대답을 세션에 저장하여 Slow LLM이 기록하게 함
            if call_sid:
                session = CallSession.get_session(call_sid)
                session["last_ai_response"] = full_fast_response

        except Exception as e:
            print(f"❌ Critical Error in ai_response_generator: {e}")
            traceback.print_exc()