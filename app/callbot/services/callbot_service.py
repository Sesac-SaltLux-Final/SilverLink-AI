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
from twilio.rest import Client as TwilioClient
from datetime import datetime

# Disable Mem0 Telemetry to prevent PostHog connection errors
os.environ["MEM0_TELEMETRY"] = "false"

from pydantic import BaseModel, Field
from openai import AsyncOpenAI
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
    acknowledgment: str = Field(description="공감 문장. 짧고 간결하게.")
    question: str = Field(description="단 하나의 질문.")
    next_action: str = Field(description="'DEEP_DIVE' 또는 'SLOT_QUESTION'")
    topic: Optional[str] = Field(description="현재 주제")

class UnifiedAnalysisResult(BaseModel):
    extracted_slots: List[SlotItem] = Field(description="발견된 정보 리스트. 없으면 빈 리스트 [] 반환.")
    dialogue_decision: DialogueDecision = Field(description="대화 전략")

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
                mem0_config = {
                    "vector_store": {"provider": "qdrant", "config": {"path": "./mem_db", "collection_name": "silverlink_memories"}},
                    "embedder": {"provider": "huggingface", "config": {"model": EMBEDDING_MODEL_NAME}}
                }
                self.memory = Memory.from_config(mem0_config)
                print("✅ Mem0 Memory Ready.")
            except Exception as e:
                print(f"⚠️ Memory Load Failed: {e}")

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
                "history": []
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
        super().__init__(callbot_repository)
        
    def test(self):
        print('test')
        
    async def build_greeting_gather_twiml(self, call_sid: str, elderly_id: str = None, elderly_name: str = None, phone_number: str = None):
        # Reset Session
        CallSession.clear_session(call_sid)
        session = CallSession.get_session(call_sid)
        session["elderly_id"] = elderly_id
        session["elderly_name"] = elderly_name

        # [New] Start Call in Backend to get call_id
        if elderly_id:
            try:
                # Phone number is required by backend. If None, we might use a dummy or skip.
                # Assuming phone_number is passed from controller.
                p_num = phone_number if phone_number else "unknown"
                call_id = await self._send_start_call_to_backend(elderly_id, elderly_name, p_num)
                if call_id:
                    session["call_id"] = call_id
                    print(f"✅ [Call Start] Assigned Call ID: {call_id}")
                else:
                    print("⚠️ [Call Start] Failed to get Call ID from Backend.")
            except Exception as e:
                print(f"❌ [Call Start] Backend Error: {e}")
        
        name_part = f"{elderly_name}님 " if elderly_name else ""
        greeting = f"안녕하세요! {name_part}실버링크에서 연락드렸습니다. 잘 지내시죠?"

        # [New] Save First Greeting to Backend
        if call_id:
            try:
                # 첫 인사말 저장 (비동기 대신 await로 순서 보장 추천, 하지만 성능상 create_task도 가능. 
                # 여기선 순서가 중요하므로 await를 고려하거나, 서버가 타임스탬프로 정렬하길 기대)
                # 안전하게 await로 저장 후 진행하거나, create_task로 던짐.
                # 에러 메시지("연결할 발화 없음")를 피하려면 어르신 답변보다 이게 먼저 DB에 들어가야 함.
                await self._send_message_to_backend(call_id, "CALLBOT", greeting)
                print(f"✅ [Call Start] Saved initial greeting to backend.")
            except Exception as e:
                print(f"⚠️ [Call Start] Failed to save greeting: {e}")

        encoded_greeting = urllib.parse.quote(greeting)
        
        # Initial greeting is pure TTS
        stream_url = f"{configs.CALL_CONTROLL_URL}/api/callbot/stream_response?text={encoded_greeting}&amp;call_sid={call_sid}&amp;mode=tts&amp;elderly_id={elderly_id}"

        twiml = f"""
        <Response>
            <Gather input="speech" action="/api/callbot/gather?elderly_id={elderly_id}" method="POST" language="ko-KR" speechTimeout="auto" bargeIn="true">
                <Play contentType="audio/basic">{stream_url}</Play>
            </Gather>
        </Response>
        """
        return twiml
        
    def make_call(self, elderly_id: int, phone_number: str, elderly_name: str):
        return self.call.calling(elderly_id, phone_number, elderly_name)

    # --- Backend Communication (Token based) ---

    async def _login_backend(self):
        """로그인하여 Access Token을 발급받고 configs를 업데이트"""
        admin_id = os.getenv('ADMIN_ID', 'admin01')
        admin_pw = os.getenv('ADMIN_PW', 'admin01')
        
        url = f"{configs.SPRING_BOOT_URL}/api/auth/login"
        payload = {
            "loginId": admin_id,
            "password": admin_pw
        }
        
        print(f"🔑 [Backend] Logging in as '{admin_id}' to {url}...")
        res = await send_data_to_backend(url, payload)
        
        if res and isinstance(res, dict) and "accessToken" in res:
            access_token = res["accessToken"]
            configs.SPRING_BOOT_API_TOKEN = access_token
            print("✅ [Backend] Login Successful! Access Token acquired.")
            return True
        else:
            print(f"❌ [Backend] Login Failed. Response: {res}")
            return False

    async def _get_auth_headers(self):
        """인증 헤더 생성. 토큰이 없으면 로그인을 시도."""
        if not configs.SPRING_BOOT_API_TOKEN:
            await self._login_backend()
            
        headers = {"Content-Type": "application/json"}
        if configs.SPRING_BOOT_API_TOKEN:
            headers["Authorization"] = f"Bearer {configs.SPRING_BOOT_API_TOKEN}"
        return headers

    async def _call_backend_api(self, url: str, payload: dict, method: str = "POST"):
        """백엔드 API 호출을 담당하며, 401 에러 시 자동으로 재로그인 후 1회 재시도합니다."""
        headers = await self._get_auth_headers()
        res = await send_data_to_backend(url, payload, method=method, headers=headers)
        
        # 401 Unauthorized Error Handling
        if res and isinstance(res, dict) and res.get("error") == "UNAUTHORIZED_401":
            print(f"⚠️ [Backend] Token expired (401). Retrying login...")
            configs.SPRING_BOOT_API_TOKEN = None # Clear token
            await self._login_backend() # Re-login
            
            headers = await self._get_auth_headers() # Get new headers
            print(f"🔄 [Backend] Retrying request to {url}...")
            res = await send_data_to_backend(url, payload, method=method, headers=headers)
        
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
        print(f"🚀 [Call Start] Requesting Call Creation: {payload}")
        res = await self._call_backend_api(url, payload)
        print(f"📥 [Call Start] Backend Response: {res}")
        
        if res and "data" in res and "callId" in res["data"]:
            return res["data"]["callId"]
        return None

    async def _send_message_to_backend(self, call_id: int, speaker: str, content: str, danger: bool = False, danger_reason: str = None):
        """대화 메시지 저장 API 호출 (CALLBOT 또는 ELDERLY)"""
        if not call_id: return
        url = f"{configs.SPRING_BOOT_URL}/api/internal/callbot/calls/{call_id}/messages"
        payload = {
            "speaker": speaker,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "danger": danger,
            "dangerReason": danger_reason
        }
        print(f"📤 [_send_message_to_backend] Sending to {url}...")
        await self._call_backend_api(url, payload)

    async def _send_end_call_to_backend(self, call_id: int, duration: int, summary: str, emotion: str, daily_status: dict, recording_url: str = None):
        """통화 종료 API 호출 (종합 데이터 포함)"""
        if not call_id: return
        url = f"{configs.SPRING_BOOT_URL}/api/internal/callbot/calls/{call_id}/end"
        
        payload = {
            "callTimeSec": duration,
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
        if len(clean_text) <= 5: return "GENERAL"
        
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
        
        # [Updated] Use call_id if available (registered at start)
        call_id = session.get("call_id")
        print(f"🕵️ [Process Conversation] CallSID: {call_sid} | Session CallID: {call_id}")
        
        # 1. PII Filtering
        timeouts['stt_processing'] = 'External (Twilio)'
        t_pii_start = time.time()
        user_input = await self.anonymize_text_async(raw_user_input)
        timeouts['pii_filtering'] = time.time() - t_pii_start
        
        # [New] Memory Retrieval
        t_mem_search_start = time.time()
        relevant_memories_text = "No relevant memories."
        if orchestrator_engine.memory:
            try:
                # Search memory asynchronously
                mem_results = await asyncio.to_thread(
                    orchestrator_engine.memory.search, 
                    user_input, 
                    user_id=call_sid, 
                    limit=3
                )
                if mem_results:
                    relevant_memories_text = "\n".join([f"- {m['memory']}" for m in mem_results])
            except Exception as e:
                print(f"Memory Search Error: {e}")
        timeouts['memory_search'] = time.time() - t_mem_search_start
        
        # 2. Intent Classification
        t_intent_start = time.time()
        intent = await self.get_intent_async(user_input)
        timeouts['intent_check'] = time.time() - t_intent_start
        
        # [Updated] Send User Message to Backend
        if call_id:
            asyncio.create_task(self._send_message_to_backend(call_id, "ELDERLY", raw_user_input, danger=(intent=="EMERGENCY")))
        
        if intent == "EMERGENCY":
            final_response = "어르신, 지금 바로 119에 도움을 요청하겠습니다! 잠시만 기다려주세요."
            
            # Update History for Emergency
            if "history" not in session: session["history"] = []
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
        
        exit_keywords = ["그만", "다음", "됐어", "종료", "아니"]
        clean_input = user_input.strip()
        force_slot_question = any(k in clean_input for k in exit_keywords) or (session["deep_dive_count"] >= MAX_DEEP_DIVE_TURNS)
        
        if not current_missing:
            force_slot_question = False
            target_slot = "작별 인사 및 건강 당부"

        unified_system_prompt = f"""
    Role: Elderly Care AI (실버링크).
    
    [Long-term Memory (Previous Conversations)]
    {relevant_memories_text}
    
    [Task 1: Information Extraction]
    Extract slots into `extracted_slots`. Categories: {MANDATORY_SLOTS}
    * CRITICAL RULES:
    1. If the user's input answers the current 'Target' question, YOU MUST extract it into that category.
    2. Keywords Mapping:
       - "밥", "식사", "먹었어", "배불러" -> [식사 여부]
       - "아파", "쑤셔", "약", "병원" -> [건강 상태]
       - "좋아", "슬퍼", "우울해", "살기 싫어" -> [기분]
       - "잤어", "못 잤어", "잠" -> [수면 상태]
       - "갈거야", "할거야", "복지관", "경로당" -> [하루 일정]
    3. Do NOT misclassify "밥 먹었다" as "기분". It is "식사 여부".
    
    [Task 2: Dialogue Generation]
    Generate a warm, short response (under 50 chars).
    - Guidelines: Emotional Support, Contextual Awareness, Polite & Friendly (Haeyo-che).
    - Context: Target="{target_slot}", Force="{force_slot_question}"
    - Use 'Long-term Memory' to personalize the conversation if relevant.
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
            
            final_response = f"{result.dialogue_decision.acknowledgment} {result.dialogue_decision.question}"

            # [Updated] Send Bot Message to Backend
            if call_id:
                asyncio.create_task(self._send_message_to_backend(call_id, "CALLBOT", final_response))    
            
            # Update Slots
            for item in result.extracted_slots:
                if item.value and str(item.value).lower() not in ["null", "none", "없음"]:
                    session["slots"][item.category] = item.value

            # Update Logic State
            if target_slot in session["slots"] and session["slots"][target_slot] is not None:
                session["deep_dive_count"] += 1
            elif result.dialogue_decision.next_action == "DEEP_DIVE" and session["deep_dive_count"] > 0:
                session["deep_dive_count"] += 1
            else:
                session["deep_dive_count"] = 0
            
            session["current_topic"] = result.dialogue_decision.topic
            
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
                "response": final_response,
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
        if not history: return "대화 내용 없음."
        
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

    async def _analyze_sentiment_with_llm(self, text: str) -> str:
        """Analyzes sentiment (GOOD, BAD, NORMAL) using LLM."""
        if not text: return "NORMAL"
        
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
            return "NORMAL"
        except Exception as e:
            print(f"Sentiment Analysis Error: {e}")
            return "NORMAL"

    async def _analyze_meal_status_with_llm(self, text: str) -> bool:
        """Analyzes meal status (True/False) using LLM."""
        if not text: return False
        
        prompt = f"""
        Determine if the user has eaten a meal based on the text.
        Text: "{text}"
        
        If they ate (or are full), output "TRUE".
        If they did not eat (or skipped), output "FALSE".
        """
        try:
            response = await self.llm_client.aclient.chat.completions.create(
                model=configs.INFERENCE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5,
                temperature=0.0
            )
            result = response.choices[0].message.content.strip().upper()
            return "TRUE" in result
        except Exception as e:
            print(f"Meal Analysis Error: {e}")
            return False

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
        if not history: return "NORMAL"
        
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

    async def _upload_recordings(self, call_sid: str) -> Optional[str]:
        def _sync_upload():
            try:
                twilio_client = TwilioClient(configs.TWILIO_SID, configs.TWILIO_TOKEN)
                recordings = twilio_client.recordings.list(call_sid=call_sid, limit=1)
                
                if not recordings:
                    print(f"⚠️ [Recording] No recordings found for {call_sid}")
                    return None

                record = recordings[0]
                recording_sid = record.sid
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
                    print(f"📤 [Recording] Uploading to S3: {configs.AWS_S3_BUCKET_NAME}/{file_key}")
                    
                    s3_client.put_object(
                        Bucket=configs.AWS_S3_BUCKET_NAME,
                        Key=file_key,
                        Body=response.content,
                        ContentType="audio/mpeg"
                    )
                    print(f"✅ [Recording] Upload success: {file_key}")
                    return file_key
                else:
                    print(f"❌ [Recording] Failed to download: {response.status_code}")
                    return None
            except Exception as e:
                print(f"❌ [Recording] Error uploading: {e}")
                traceback.print_exc()
                return None
        
        return await asyncio.to_thread(_sync_upload)

    async def finalize_call(self, call_sid: str, duration: str = "0"):
        """Called when Twilio call status is 'completed' to save history."""
        session = CallSession.get_session(call_sid)
        history = session.get("history", [])
        call_id = session.get("call_id")
        
        # [New] Upload Recording
        recording_key = await self._upload_recordings(call_sid)
        
        if history:
            # 1. Save to Long-term Memory (Mem0)
            if orchestrator_engine.memory:
                print(f"📞 [Call End] Triggering batch save for {len(history)} turns.")
                await self._save_full_history_async(call_sid, history)
            
            # 2. Finalize to Backend
            if call_id:
                print(f"📝 [Call End] Finalizing backend data for call {call_id}...")
                summary = await self._summarize_conversation(history)
                daily_status = await self._map_slots_to_daily_status(session.get("slots", {}))
                emotion = await self._analyze_overall_emotion(history)
                
                try: duration_sec = int(duration)
                except: duration_sec = 0
                
                await self._send_end_call_to_backend(call_id, duration_sec, summary, emotion, daily_status, recording_key)
                print("✅ [Backend] Call Finalized successfully.")
        
        # Finally clear the session
        CallSession.clear_session(call_sid)
        print(f"🧹 [Call End] Session cleared for {call_sid}")

    async def _save_full_history_async(self, user_id: str, history: List[Dict]):
        """Helper to save full history to Mem0 in background"""
        if not orchestrator_engine.memory: return
        
        def _batch_save():
            for turn in history:
                try:
                    orchestrator_engine.memory.add(
                        f"사용자: {turn['user']} | 상담사: {turn['ai']}", 
                        user_id=user_id
                    )
                except Exception as e:
                    print(f"Mem0 Save Error: {e}")
        
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

    # --- Streaming Logic (Modified for TTS only) ---
    async def ai_response_generator(self, text: str, history: List[dict], mode: str = "chat", start_ts: float = 0.0, elderly_id: str = None) -> AsyncGenerator[bytes, None]:
        """
        Sentence-level TTS Streaming: Splits text and streams audio for each sentence immediately.
        """
        if not text: return
        
        try:
            sentences = re.split(r'(?<=[.!?])\s*', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if not sentences:
                sentences = [text]
                
            print(f"🔊 [TTS Streaming] Processing {len(sentences)} sentences from text len {len(text)}")
            
            for i, sentence in enumerate(sentences):
                if len(sentence) < 2 and sentence in [".", "!", "?", ","]:
                    continue
                    
                t_tts_start = time.time()
                try:
                    wav_data = await self.generate_tts_stream(sentence)
                    tts_duration = time.time() - t_tts_start
                    
                    if wav_data:
                        print(f"🔊 [TTS Gen {i+1}/{len(sentences)}]: {tts_duration:.3f}s | {len(wav_data)} bytes | '{sentence[:20]}...' ")
                        
                        if i == 0 and start_ts > 0:
                            first_byte_latency = time.time() - start_ts
                            print(f"⏱️ [Latency] Start to First Audio: {first_byte_latency:.3f}s")
                        
                        ulaw_data = self.wav_to_ulaw(wav_data)
                        if ulaw_data:
                            yield ulaw_data
                        else:
                            print(f"⚠️ WAV to ULAW conversion failed for sentence {i+1}")
                    else:
                        print(f"⚠️ TTS Generation failed (empty) for sentence {i+1}")
                        
                except Exception as e:
                    print(f"❌ Error processing sentence {i+1}: {e}")
                    continue

            if start_ts > 0:
                total_latency = time.time() - start_ts
                print(f"⏱️ [Latency] Total Process Time: {total_latency:.3f}s")
                
        except Exception as e:
            print(f"❌ Critical Error in ai_response_generator: {e}")
            traceback.print_exc()