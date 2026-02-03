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
from app.callbot.services.callbot_service import CallbotService
from app.queue.sqs_client import SQSClient
from app.queue.message_schema import CallRequestMessage
from app.core.config import configs

# 로깅 설정
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# 콘솔 핸들러 추가 (없으면)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
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
    
@router.post("/call")
@inject_callbot
def get_call(
    request: CallScheduleRequest,
    service: CallbotService = Depends(Provide[Container.callbot_service]),
    sqs_client: SQSClient = Depends(Provide[Container.sqs_client])
):
    logger.info("📞 [GET /callbot/call] 전화 걸기 요청")
    try:
        # SQS
        ####################################################
        message = CallRequestMessage(
            message_id=str(uuid.uuid4()),
            # schedule_id=request.schedule_id,
            elderly_id=request.elderly_id,
            elderly_name=request.elderly_name,
            phone_number=request.phone_number,
            # scheduled_time=request.scheduled_time,
            retry_count=0
        )
        
        message_id = sqs_client.publish(message)
        if message_id:
            logger.info("✅ [POST /callbot/schedule-call] SQS 발행 성공")
            logger.info("="*50)
        #####################################################
        result = service.make_call(request.elderly_id,request.phone_number,request.elderly_name)
        logger.info("✅ [GET /callbot/call] 전화 걸기 성공")
        return result
    except Exception as e:
        logger.error(f"❌ [GET /callbot/call] 에러 발생: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

    
@router.api_route("/voice", methods=["POST"])
@inject_callbot
async def voice(
    request: Request,
    service: CallbotService = Depends(Provide[Container.callbot_service])
):
    """전화가 처음 연결되었을 때 실행"""
    logger.info("="*50)
    logger.info("📞 [POST /callbot/voice] Voice 엔드포인트 호출됨")
    
    try:
        # 1. Form 데이터 파싱
        logger.debug("1️⃣ Form 데이터 및 쿼리 파라미터 파싱 중...")
        form_data = await request.form()
        call_sid = form_data.get("CallSid", "unknown")
        phone_number = form_data.get("To")
        
        # 쿼리 파라미터에서 데이터 추출 (elderly_id, elderly_name)
        elderly_id = request.query_params.get("elderly_id")
        elderly_name = request.query_params.get("elderly_name")
        
        # 2. 대화 히스토리 초기화
        logger.debug("2️⃣ 대화 히스토리 초기화...")
        conversation_history[call_sid] = []
        
        # [Updated] await call
        twiml = await service.build_greeting_gather_twiml(
            call_sid=call_sid, 
            elderly_id=elderly_id, 
            elderly_name=elderly_name,
            phone_number=phone_number
        )
        return Response(content=twiml, media_type="application/xml")
        
    except Exception as e:
        logger.error("="*50)
        logger.error("❌ [POST /callbot/voice] 에러 발생!")
        logger.error(f"에러 타입: {type(e).__name__}")
        logger.error(f"에러 메시지: {e}")
        logger.error(f"스택 트레이스:\n{traceback.format_exc()}")
        logger.error("="*50)
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
            service.ai_response_generator(text, history, mode, start_ts, elderly_id),
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
    service: CallbotService = Depends(Provide[Container.callbot_service])
):
    """Twilio SpeechResult Handler with Orchestrator Logic"""
    # logger.info("🎤 [POST /callbot/gather] Gather 엔드포인트 호출")
    
    try:
        form_data = await request.form()
        speech_result = form_data.get("SpeechResult")
        call_sid = form_data.get("CallSid", "unknown")
        
        # logger.debug(f"   CallSid: {call_sid}")
        # logger.debug(f"   SpeechResult: {speech_result}")
            
        if not speech_result:
            twiml = """
            <Response>
                <Gather input="speech" action="/api/callbot/gather" method="POST" language="ko-KR" speechTimeout="auto">
                </Gather>
            </Response>
            """
            return Response(content=twiml, media_type="application/xml")

        # 쿼리 파라미터에서 데이터 추출 (elderly_id, elderly_name)
        elderly_id = request.query_params.get("elderly_id")
        # elderly_name = request.query_params.get("elderly_name")
        
        logger.info(f"🎤 사용자 발화: {speech_result}")
        
        # --- Orchestrator Logic Call ---
        start_time = datetime.now()
        result = await service.process_conversation(call_sid,elderly_id, speech_result)
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"⚡ 처리 완료 ({duration:.3f}s): Intent={result.get('intent')}")
        
        response_text = result.get("response", "죄송합니다, 다시 말씀해 주시겠어요?")
        
        # 응급 상황 처리
        if result.get("intent") == "EMERGENCY":
            logger.critical(f"🚨 EMERGENCY DETECTED: {call_sid}")
            # 응급 상황 시 즉시 안내 후 종료하거나 상담원 연결
            encoded_text = urllib.parse.quote(response_text)
            stream_url = f"{configs.CALL_CONTROLL_URL}/api/callbot/stream_response?text={encoded_text}&amp;call_sid={call_sid}&amp;mode=tts&amp;elderly_id={elderly_id}"
            
            twiml = f"""
            <Response>
                <Play contentType="audio/basic">{stream_url}</Play>
                <Pause length="1"/>
                <Hangup/>
            </Response>
            """
            return Response(content=twiml, media_type="application/xml")
        
        # 일반 대화 처리
        encoded_text = urllib.parse.quote(response_text)
        current_ts = datetime.now().timestamp()
        stream_url = f"{configs.CALL_CONTROLL_URL}/api/callbot/stream_response?text={encoded_text}&amp;call_sid={call_sid}&amp;mode=tts&amp;start_ts={current_ts}&amp;elderly_id={elderly_id}"

        twiml = f"""
        <Response>
            <Gather input="speech" action="/api/callbot/gather?elderly_id={elderly_id}" method="POST" language="ko-KR" speechTimeout="auto" bargeIn="true">
                <Play contentType="audio/basic">{stream_url}</Play>
            </Gather>
        </Response>
        """
        return Response(content=twiml, media_type="application/xml")
        
    except Exception as e:
        logger.error(f"❌ [POST /callbot/gather] 에러 발생: {e}")
        logger.error(traceback.format_exc())
        
        # 에러 발생 시 안전하게 다시 묻기
        error_msg = urllib.parse.quote("죄송해요, 잠시 문제가 생겼어요. 다시 말씀해 주시겠어요?")
        stream_url = f"{configs.CALL_CONTROLL_URL}/api/callbot/stream_response?text={error_msg}&amp;call_sid={call_sid}&amp;mode=tts"
        twiml = f"""
        <Response>
            <Gather input="speech" action="/api/callbot/gather" method="POST" language="ko-KR" speechTimeout="auto" bargeIn="true">
                <Play contentType="audio/basic">{stream_url}</Play>
            </Gather>
        </Response>
        """
        return Response(content=twiml, media_type="application/xml")
