import logging
import traceback
from typing import Dict, List
from datetime import datetime
import uuid

import urllib.parse
from dependency_injector.wiring import Provide
from fastapi import APIRouter, Depends, Request, HTTPException
from fastapi import BackgroundTasks, Form
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field, ConfigDict

from app.core.container import Container
from app.core.middleware import inject_callbot
from app.callbot.services.callbot_service import CallbotService, CallSession
from app.queue.sqs_client import SQSClient
from app.queue.message_schema import CallRequestMessage
from app.core.config import configs

# 로깅 설정
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 콘솔 핸들러 추가 (없으면)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

router = APIRouter(
    prefix="/callbot",
    tags=["callbot"],
)

# Global In-Memory History (Key: CallSid, Value: List[dict])
# Note: In a production environment with multiple instances, use Redis or a Database.
conversation_history: Dict[str, List[dict]] = {}


# Request Schema for SQS Call Scheduling
class CallScheduleRequest(BaseModel):
    """통화 스케줄 요청 스키마"""
    # schedule_id: int = Field(..., description="통화 스케줄 ID")
    elderly_id: int = Field(..., description="어르신 ID")
    elderly_name: str = Field(..., description="어르신 이름")
    phone_number: str = Field(..., description="전화번호 (E.164 형식: +821012345678)")
    # scheduled_time: datetime = Field(..., description="예약된 통화 시간")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "elderly_id": 100,
                "elderly_name": "홍길동",
                "phone_number": "+821012345678",
            }
        }
    )

@router.get("")
@inject_callbot
def get_post_list(
    service: CallbotService = Depends(Provide[Container.callbot_service]),
):
    logger.info("🔍 [GET /callbot] 테스트 엔드포인트 호출")
    try:
        result = service.test()
        logger.info("✅ [GET /callbot] 테스트 완료")
        return result
    except Exception as e:
        logger.error(f"❌ [GET /callbot] 에러 발생: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/memory/{elderly_id}")
@inject_callbot
def get_memory(
    elderly_id: int,
    service: CallbotService = Depends(Provide[Container.callbot_service])
):
    """특정 어르신의 장기 기억 조회"""
    try:
        memories = service.get_memories(elderly_id)
        return {"elderly_id": elderly_id, "memories": memories}
    except Exception as e:
        logger.error(f"❌ [GET /memory] Error: {e}")
        return {"error": str(e)}

@router.post("/call")
@inject_callbot
async def get_call(
    request: CallScheduleRequest,
    service: CallbotService = Depends(Provide[Container.callbot_service]),
    sqs_client: SQSClient = Depends(Provide[Container.sqs_client])
):
    logger.info("📞 [POST /callbot/call] 전화 걸기 요청")
    try:
        # SQS 발행 (백그라운드 처리용)
        ####################################################
        message = CallRequestMessage(
            message_id=str(uuid.uuid4()),
            elderly_id=request.elderly_id,
            elderly_name=request.elderly_name,
            phone_number=request.phone_number,
            retry_count=0
        )
        
        message_id = sqs_client.publish(message)
        if message_id:

            logger.info("✅ [POST /callbot/call] SQS 발행 성공")
            logger.info("="*50)
        #####################################################
        
        # 통화 시작 (Pre-compute greeting + Twilio 연결)
        result = await service.make_call(
            request.elderly_id,
            request.phone_number,
            request.elderly_name
        )
        logger.info(f"✅ [POST /callbot/call] 전화 걸기 성공: call_sid={result.get('call_sid')}")
        return result
    except Exception as e:
        logger.error(f"❌ [POST /callbot/call] 에러 발생: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

    
@router.api_route("/voice", methods=["POST"])
@inject_callbot
async def voice(
    request: Request,
    service: CallbotService = Depends(Provide[Container.callbot_service])
):
    """전화가 처음 연결되었을 때 실행"""
    logger.info("📞 [POST /callbot/voice] Voice 엔드포인트 호출됨")
    
    try:
        # 1. Form 데이터 파싱
        logger.debug("1️⃣ Form 데이터 및 쿼리 파라미터 파싱 중...")
        form_data = await request.form()
        call_sid = form_data.get("CallSid", "unknown")
        phone_number = form_data.get("To")
        
        # 쿼리 파라미터에서 데이터 추출 (elderly_id, elderly_name, initial_mem, greeting)
        elderly_id = request.query_params.get("elderly_id")
        elderly_name = request.query_params.get("elderly_name")
        initial_mem = request.query_params.get("initial_mem", "")
        greeting = request.query_params.get("greeting", "")
        
        # 2. 대화 히스토리 초기화
        logger.debug("2️⃣ 대화 히스토리 초기화...")
        conversation_history[call_sid] = []
        
        # [Updated] await call
        twiml = await service.build_greeting_gather_twiml(
            call_sid=call_sid, 
            elderly_id=elderly_id, 
            elderly_name=elderly_name,
            phone_number=phone_number,
            initial_mem=initial_mem,
            greeting=greeting
        )
        return Response(content=twiml, media_type="application/xml")
        
    except Exception as e:
        logger.error("❌ [POST /callbot/voice] 에러 발생!")
        logger.error(f"에러 타입: {type(e).__name__}")
        logger.error(f"에러 메시지: {e}")
        logger.error(f"스택 트레이스:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Voice 처리 실패: {str(e)}")

@router.get("/stream_response")
@inject_callbot
async def stream_response(
    text: str,
    call_sid: str = None,
    mode: str = "chat",
    start_ts: float = 0.0,
    elderly_id: str = None,
    service: CallbotService = Depends(Provide[Container.callbot_service])
):
    """Streams audio chunks dynamically generated from LLM -> TTS"""

    
    try:
        if call_sid:
            history = conversation_history.get(call_sid, [])
        else:
            history = []
        logger.debug(f"   history 길이: {len(history)}")
        
        return StreamingResponse(
            service.ai_response_generator(text, history, mode, start_ts, elderly_id, call_sid),
            media_type="audio/basic",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "X-Accel-Buffering": "no" # Disable Nginx buffering if any
            }
        )
    except Exception as e:
        logger.error(f"❌ [GET /callbot/stream_response] 에러: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    

@router.post("/s3-upload")
@inject_callbot
async def s3_upload_callback(
    background_tasks: BackgroundTasks,
    CallSid: str = Form(...),
    RecordingUrl: str = Form(...),
    RecordingSid: str = Form(...),
    RecordingStatus: str = Form(...),
    RecordingDuration: int = Form(None), # Twilio에서 보내주는 녹음 시간(초)
    service: CallbotService = Depends(Provide[Container.callbot_service])
):
    """Twilio Recording Status Callback Handler"""
    logger.info(f"📼 [S3 Upload] Callback received for Call: {CallSid}, Duration: {RecordingDuration}s, Status: {RecordingStatus}")
    
    if RecordingStatus == 'completed':
        background_tasks.add_task(service.upload_recording_from_url, RecordingUrl, RecordingSid, CallSid, RecordingDuration)
        return {"message": "Upload task started"}
    
    return {"message": "Recording not completed"}

@router.post("/status")
@inject_callbot
async def call_status(
    request: Request,
    service: CallbotService = Depends(Provide[Container.callbot_service])
):
    """Twilio Call Status Callback Handler"""
    try:
        form_data = await request.form()
        call_sid = form_data.get("CallSid")
        call_status = form_data.get("CallStatus")
        call_duration = form_data.get("CallDuration", "0") # 통화 시간(초) 추출
        
        logger.info(f"📞 [Status] {call_sid} -> {call_status} (Duration: {call_duration}s)")
        
        # 통화가 종료된 경우 정리 작업 수행
        if call_status in ["completed", "failed", "busy", "no-answer", "canceled"]:
            await service.finalize_call(call_sid, call_duration)
            
        return Response(status_code=200)
    except Exception as e:
        logger.error(f"❌ [POST /callbot/status] Error: {e}")
        return Response(status_code=500)

@router.api_route("/gather", methods=["POST"])
@inject_callbot
async def gather(
    request: Request,
    background_tasks: BackgroundTasks,
    service: CallbotService = Depends(Provide[Container.callbot_service])
):
    """Twilio SpeechResult Handler with Orchestrator Logic"""
    # logger.info("🎤 [POST /callbot/gather] Gather 엔드포인트 호출")
    
    try:
        form_data = await request.form()
        speech_result = form_data.get("SpeechResult")
        call_sid = form_data.get("CallSid", "unknown")
        
        # 쿼리 파라미터 추출
        elderly_id = request.query_params.get("elderly_id")
        retry_count = int(request.query_params.get("retry", 0)) # 재시도 횟수
        
        # [Case: No Input] 사용자가 아무 말도 안 했을 때 (None, 빈 문자열, 공백 포함)
        if not speech_result or not speech_result.strip():
            if retry_count < 1:
                # 1차 재시도: 다시 물어봄
                logger.info(f"🔇 [No Input] Retry {retry_count + 1}/2")
                retry_msg = urllib.parse.quote("죄송해요, 제가 잘 못 들었어요. 다시 한번 말씀해 주시겠어요?")
                stream_url = f"{configs.CALL_CONTROLL_URL}/api/callbot/stream_response?text={retry_msg}&amp;call_sid={call_sid}&amp;mode=tts"
                
                hints = "밥, 식사, 아침, 점심, 저녁, 건강, 아파, 병원, 약, 기분, 좋아, 우울해, 심심해, 산책, 운동, 복지관, 노인정, 잠, 주무셨어, 꿈, 어르신, 안녕, 응, 그래, 아니"
                twiml = f"""
                <Response>
                    <Gather input="speech" action="/api/callbot/gather?elderly_id={elderly_id}&amp;retry={retry_count + 1}" method="POST" language="ko-KR" speechTimeout="2.0" bargeIn="true" enhanced="true" hints="{hints}" profanityFilter="false">
                        <Play contentType="audio/basic">{stream_url}</Play>
                    </Gather>
                </Response>
                """
                return Response(content=twiml, media_type="application/xml")
            else:
                # 2차 무응답: 종료 안내 멘트 후 끊기
                logger.info("🔇 [No Input] Max retries reached. Hanging up.")
                bye_msg = urllib.parse.quote("답변이 없으셔서 통화를 종료할게요. 다음에 또 연락드릴게요. 건강하세요!")
                stream_url = f"{configs.CALL_CONTROLL_URL}/api/callbot/stream_response?text={bye_msg}&amp;call_sid={call_sid}&amp;mode=tts"
                
                twiml = f"""
                <Response>
                    <Play contentType="audio/basic">{stream_url}</Play>
                    <Hangup/>
                </Response>
                """
                return Response(content=twiml, media_type="application/xml")

        # [Case: Valid Input] 사용자가 말을 했을 때 (retry 초기화)
        logger.info(f"🎤 사용자 발화: {speech_result}")
        
        # [Immediate Exit Check] 종료 키워드 감지 (FastAPI 레벨에서 즉시 처리)
        exit_keywords = ["그만", "그만해", "됐어", "종료", "끊어", "끊을게", "다음에하자", "또전화", "다음에연락", "들어가세요", "고마워요"]
        clean_input = speech_result.replace(" ", "")
        
        # 종료 감지 로직 (예외 처리 추가: '끊어졌어' 등)
        is_exit = any(kw in clean_input for kw in exit_keywords)
        if "끊어졌어" in clean_input: # '끊어'가 들어있지만 예외인 경우
            is_exit = False
            
        if is_exit:
            logger.info(f"🛑 [Immediate Exit] User requested termination: {speech_result}")
            
            # 작별 인사말 (캐시된 문구와 100% 일치해야 함)
            bye_text = "네, 알겠습니다. 어르신, 편히 쉬시고 다음에 또 목소리 들려주세요. 건강하세요!"
            
            # 중요: 즉시 종료하더라도 서비스 로직을 호출하여 데이터 저장 및 분석을 수행하게 함
            background_tasks.add_task(service.process_conversation, call_sid, elderly_id, speech_result)
            
            bye_msg_encoded = urllib.parse.quote(bye_text)
            stream_url = f"{configs.CALL_CONTROLL_URL}/api/callbot/stream_response?text={bye_msg_encoded}&amp;call_sid={call_sid}&amp;mode=tts"
            
            twiml = f"""
            <Response>
                <Play contentType="audio/basic">{stream_url}</Play>
                <Hangup/>
            </Response>
            """
            return Response(content=twiml, media_type="application/xml")
        
        # [Immediate Emergency Check] 응급 상황 감지
        emergency_keywords = ["살려줘", "숨이 안", "숨 못", "가슴이 너무 아파", "쓰러졌", "119", "죽을 것 같", "도와줘", "큰일났어"]
        if any(k in speech_result for k in emergency_keywords):
            logger.info(f"🚨 [Immediate Emergency] Emergency detected: {speech_result}")
            emergency_msg = urllib.parse.quote("어르신 확인했습니다. 안전을 위해 담당 상담사님과 보호자님께 긴급알림을 즉시 전송하겠습니다.")
            stream_url = f"{configs.CALL_CONTROLL_URL}/api/callbot/stream_response?text={emergency_msg}&amp;call_sid={call_sid}&amp;mode=tts"
            
            # 백그라운드 작업으로 긴급 상황 처리 (Backend 전송 등) 실행
            background_tasks.add_task(service.process_conversation, call_sid, elderly_id, speech_result)
            
            twiml = f"""
            <Response>
                <Play contentType="audio/basic">{stream_url}</Play>
                <Hangup/>
            </Response>
            """
            return Response(content=twiml, media_type="application/xml")
        
        # [Ultra-Fast Response Strategy]
        # 1. 즉시 스트리밍 URL 생성 (LLM 대기 없음)
        #    - text 파라미터에 '사용자 발화'를 그대로 넣어서 보냄 (서비스에서 이걸 보고 LLM 생성)
        encoded_input = urllib.parse.quote(speech_result)
        current_ts = datetime.now().timestamp()
        
        # [Updated] 대화 히스토리 업데이트 (User 발화 추가)
        if call_sid not in conversation_history:
            conversation_history[call_sid] = []
        conversation_history[call_sid].append({"user": speech_result, "ai": ""}) # AI 답변은 나중에 채워짐

        stream_url = f"{configs.CALL_CONTROLL_URL}/api/callbot/stream_response?text={encoded_input}&amp;call_sid={call_sid}&amp;mode=chat&amp;start_ts={current_ts}&amp;elderly_id={elderly_id}"

        # 2. 분석 및 저장은 백그라운드에서 천천히 수행
        #    (슬롯 필링, 감정 분석, DB 저장 등)
        background_tasks.add_task(service.process_conversation, call_sid, elderly_id, speech_result)

        # 3. TwiML 즉시 반환
        # [마무리 체크] 현재 목표가 '작별 인사'인 경우에만 전화를 끊도록 설정
        session = CallSession.get_session(call_sid)
        missing_slots = [k for k, v in session.get("slots", {}).items() if v is None]
        current_target = missing_slots[0] if missing_slots else "작별 인사 및 건강 당부"
        
        # 진짜 마지막 단계(작별 인사)일 때만 끊기
        is_finish = (current_target == "작별 인사 및 건강 당부")
        
        hints = "밥, 식사, 아침, 점심, 저녁, 건강, 아파, 병원, 약, 기분, 좋아, 우울해, 심심해, 산책, 운동, 우동,전복 죽,김치, 노인정,허리, 잠, 주무셨어, 꿈, 어르신, 안녕, 응, 그래, 아니"

        if is_finish:
            twiml = f"""
            <Response>
                <Play contentType="audio/basic">{stream_url}</Play>
                <Pause length="5"/>
                <Hangup/>
            </Response>
            """
        else:
            twiml = f"""
            <Response>
                <Gather input="speech" action="/api/callbot/gather?elderly_id={elderly_id}" method="POST" language="ko-KR" speechTimeout="auto" bargeIn="true" timeout="5" speechModel="phone_call" enhanced="true" hints="{hints}" profanityFilter="false">
                    <Play contentType="audio/basic">{stream_url}</Play>
                </Gather>
                <Redirect>/api/callbot/gather?elderly_id={elderly_id}&amp;retry=0</Redirect>
            </Response>
            """
        return Response(content=twiml, media_type="application/xml")
        
    except Exception as e:
        logger.error(f"❌ [POST /callbot/gather] 에러 발생: {e}")
        logger.error(traceback.format_exc())
        
        # 에러 발생 시 안전하게 다시 묻기
        error_msg = urllib.parse.quote("죄송해요, 잠시 문제가 생겼어요. 다시 말씀해 주시겠어요?")
        stream_url = f"{configs.CALL_CONTROLL_URL}/api/callbot/stream_response?text={error_msg}&amp;call_sid={call_sid}&amp;mode=tts"
        
        hints = "밥, 식사, 아침, 점심, 저녁, 건강, 아파, 병원, 약, 기분, 좋아, 우울해, 심심해, 산책, 운동, 복지관, 노인정, 잠, 주무셨어, 꿈, 어르신, 안녕, 응, 그래, 아니"
        twiml = f"""
        <Response>
            <Gather input="speech" action="/api/callbot/gather" method="POST" language="ko-KR" speechTimeout="2.0" bargeIn="true" enhanced="true" hints="{hints}" profanityFilter="false">
                <Play contentType="audio/basic">{stream_url}</Play>
            </Gather>
        </Response>
        """
        return Response(content=twiml, media_type="application/xml")
