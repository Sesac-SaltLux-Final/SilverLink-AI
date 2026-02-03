"""
SQS Worker

SQS 메인 큐에서 메시지를 폴링하고 처리하는 워커 클래스.
기존 CallbotService를 활용하여 통화를 실행합니다.
"""
import asyncio
import logging
import signal
from typing import Optional
from datetime import datetime

from app.queue.sqs_client import SQSClient
from app.queue.message_schema import CallRequestMessage, CallResultMessage, CallStatus
from app.callbot.services.callbot_service import CallbotService
from app.integration.call import CALL

logger = logging.getLogger(__name__)


class SQSWorker:
    """
    SQS 메시지 처리 워커
    
    메인 큐에서 통화 요청을 폴링하고, CallbotService를 통해 처리합니다.
    """
    
    MAX_RETRY_COUNT = 3  # 최대 재시도 횟수
    
    def __init__(
        self,
        sqs_client: SQSClient,
        callbot_service: Optional[CallbotService] = None,
        call_client: Optional[CALL] = None
    ):
        """
        워커 초기화
        
        Args:
            sqs_client: SQS 클라이언트
            callbot_service: CallbotService 인스턴스 (DI)
            call_client: CALL 인스턴스 (DI)
        """
        self.sqs_client = sqs_client
        self.callbot_service = callbot_service
        self.call_client = call_client
        self._running = False
        self._shutdown_event = asyncio.Event()
        
        logger.info("SQS Worker initialized")
    
    def _setup_signal_handlers(self):
        """Graceful shutdown을 위한 시그널 핸들러 설정"""
        def handle_signal(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            self._running = False
            self._shutdown_event.set()
        
        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)
    
    async def start(self, poll_interval: int = 1):
        """
        워커 시작 - 메인 폴링 루프
        
        Args:
            poll_interval: 폴링 간격 (초)
        """
        self._running = True
        self._setup_signal_handlers()
        
        logger.info("🚀 SQS Worker started, listening for messages...")
        
        while self._running:
            try:
                # Long Polling으로 메시지 수신
                messages = self.sqs_client.receive(
                    max_messages=10,
                    wait_time_seconds=20,
                    visibility_timeout=60
                )
                
                if messages:
                    # 병렬로 메시지 처리
                    tasks = [self._process_message(msg) for msg in messages]
                    await asyncio.gather(*tasks, return_exceptions=True)
                else:
                    # 메시지가 없으면 잠시 대기
                    await asyncio.sleep(poll_interval)
                    
            except Exception as e:
                logger.error(f"Error in polling loop: {e}")
                await asyncio.sleep(5)  # 에러 발생 시 대기 후 재시도
        
        logger.info("👋 SQS Worker stopped gracefully")
    
    async def stop(self):
        """워커 중지"""
        self._running = False
        self._shutdown_event.set()
        logger.info("Worker stop requested")
    
    async def _process_message(self, raw_message: dict) -> bool:
        """
        단일 메시지 처리
        
        Args:
            raw_message: SQS에서 수신한 원본 메시지
            
        Returns:
            처리 성공 여부
        """
        receipt_handle = raw_message.get('ReceiptHandle')
        message_id = raw_message.get('MessageId')
        
        try:
            # 메시지 파싱
            body = raw_message.get('Body', '{}')
            call_request = CallRequestMessage.model_validate_json(body)
            
            logger.info(f"📞 Processing call request: schedule_id={call_request.schedule_id}, "
                       f"elderly={call_request.elderly_name}, retry={call_request.retry_count}")
            
            # 통화 실행
            result = await self._execute_call(call_request)
            
            if result.status == CallStatus.COMPLETED:
                # 성공: 메시지 삭제
                self.sqs_client.delete(receipt_handle)
                logger.info(f"✅ Call completed successfully: {call_request.schedule_id}")
                return True
            else:
                # 실패: 재시도 또는 DLQ 이동
                await self._handle_failure(call_request, receipt_handle, result.error_message)
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to process message {message_id}: {e}")
            
            # 파싱 실패 등 심각한 에러는 DLQ로 바로 이동
            try:
                call_request = CallRequestMessage.model_validate_json(raw_message.get('Body', '{}'))
                self.sqs_client.move_to_dlq(call_request, str(e))
                self.sqs_client.delete(receipt_handle)
            except Exception:
                pass  # 파싱도 실패하면 무시
            
            return False
    
    async def _execute_call(self, request: CallRequestMessage) -> CallResultMessage:
        """
        실제 통화 실행
        
        Args:
            request: 통화 요청 메시지
            
        Returns:
            통화 결과 메시지
        """
        try:
            # CallbotService 또는 CALL 클라이언트를 사용하여 통화 실행
            if self.callbot_service:
                # CallbotService.make_call() 사용 (기존 로직 재사용)
                self.callbot_service.make_call()
                call_sid = "simulated_call_sid"  # 실제 구현시 Twilio 응답에서 추출
            elif self.call_client:
                # CALL 클라이언트 직접 사용
                self.call_client.calling()
                call_sid = "simulated_call_sid"
            else:
                raise ValueError("No call client configured")
            
            return CallResultMessage(
                message_id=request.message_id,
                schedule_id=request.schedule_id,
                call_sid=call_sid,
                status=CallStatus.COMPLETED,
                processed_at=datetime.utcnow()
            )
            
        except Exception as e:
            return CallResultMessage(
                message_id=request.message_id,
                schedule_id=request.schedule_id,
                status=CallStatus.FAILED,
                error_message=str(e),
                processed_at=datetime.utcnow()
            )
    
    async def _handle_failure(
        self,
        request: CallRequestMessage,
        receipt_handle: str,
        error_message: str
    ):
        """
        실패 처리 - 재시도 또는 DLQ 이동
        
        Args:
            request: 원본 요청
            receipt_handle: 메시지 핸들
            error_message: 에러 메시지
        """
        if request.retry_count < self.MAX_RETRY_COUNT:
            # 재시도: 메시지를 다시 큐에 넣음 (지연 적용)
            retry_delay = (request.retry_count + 1) * 30  # 30초, 60초, 90초 지연
            
            new_request = request.model_copy(update={'retry_count': request.retry_count + 1})
            self.sqs_client.publish(new_request, delay_seconds=retry_delay)
            self.sqs_client.delete(receipt_handle)
            
            logger.warning(f"🔄 Retry scheduled: schedule_id={request.schedule_id}, "
                          f"attempt={request.retry_count + 1}, delay={retry_delay}s")
        else:
            # 최대 재시도 초과: DLQ로 이동
            self.sqs_client.move_to_dlq(request, error_message)
            self.sqs_client.delete(receipt_handle)
            
            logger.error(f"💀 Max retries exceeded, moved to DLQ: schedule_id={request.schedule_id}")
