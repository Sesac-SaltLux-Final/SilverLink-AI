"""
Dead Letter Queue Handler

DLQ에 있는 실패 메시지를 처리하고 모니터링합니다.
"""
import json
import logging
from typing import List, Optional, Callable
from datetime import datetime

from app.queue.sqs_client import SQSClient
from app.queue.message_schema import DLQMessage, CallRequestMessage

logger = logging.getLogger(__name__)


class DLQHandler:
    """
    Dead Letter Queue 핸들러
    
    실패한 메시지를 모니터링하고 알림 처리를 담당합니다.
    """
    
    def __init__(
        self,
        sqs_client: SQSClient,
        alert_callback: Optional[Callable[[DLQMessage], None]] = None
    ):
        """
        DLQ 핸들러 초기화
        
        Args:
            sqs_client: SQS 클라이언트
            alert_callback: 알림 콜백 함수 (선택)
        """
        self.sqs_client = sqs_client
        self.alert_callback = alert_callback or self._default_alert
        
        logger.info("DLQ Handler initialized")
    
    def _default_alert(self, dlq_message: DLQMessage):
        """기본 알림 처리 - 로그 기록"""
        logger.critical(
            f"🚨 DLQ ALERT: schedule_id={dlq_message.original_message.schedule_id}, "
            f"elderly={dlq_message.original_message.elderly_name}, "
            f"reason={dlq_message.failure_reason}, "
            f"attempts={dlq_message.total_attempts}"
        )
    
    def poll_dlq(self, max_messages: int = 10) -> List[DLQMessage]:
        """
        DLQ에서 실패 메시지 조회
        
        Args:
            max_messages: 최대 조회 메시지 수
            
        Returns:
            DLQ 메시지 목록
        """
        if not self.sqs_client.dlq_url:
            logger.warning("DLQ URL not configured")
            return []
        
        # DLQ에서 메시지 수신을 위해 임시로 queue_url 스왑
        original_url = self.sqs_client.queue_url
        self.sqs_client.queue_url = self.sqs_client.dlq_url
        
        try:
            raw_messages = self.sqs_client.receive(
                max_messages=max_messages,
                wait_time_seconds=5,
                visibility_timeout=30
            )
            
            dlq_messages = []
            for msg in raw_messages:
                try:
                    body = json.loads(msg.get('Body', '{}'))
                    original_msg = CallRequestMessage.model_validate(body.get('original_message', {}))
                    
                    dlq_message = DLQMessage(
                        original_message=original_msg,
                        failure_reason=body.get('failure_reason', 'Unknown'),
                        failed_at=datetime.fromisoformat(body.get('failed_at', datetime.utcnow().isoformat())),
                        total_attempts=body.get('total_attempts', 0)
                    )
                    dlq_messages.append(dlq_message)
                    
                except Exception as e:
                    logger.error(f"Failed to parse DLQ message: {e}")
            
            return dlq_messages
            
        finally:
            # queue_url 복원
            self.sqs_client.queue_url = original_url
    
    def process_dlq_messages(self, messages: List[DLQMessage]) -> dict:
        """
        DLQ 메시지 처리 및 알림 발송
        
        Args:
            messages: DLQ 메시지 목록
            
        Returns:
            처리 결과 통계
        """
        stats = {
            'total': len(messages),
            'alerted': 0,
            'errors': 0
        }
        
        for msg in messages:
            try:
                self.alert_callback(msg)
                stats['alerted'] += 1
            except Exception as e:
                logger.error(f"Failed to send alert: {e}")
                stats['errors'] += 1
        
        logger.info(f"DLQ processing complete: {stats}")
        return stats
    
    def get_dlq_stats(self) -> dict:
        """
        DLQ 통계 조회
        
        Returns:
            DLQ 큐 속성 (메시지 수 등)
        """
        if not self.sqs_client.dlq_url:
            return {'error': 'DLQ URL not configured'}
        
        # DLQ 속성 조회를 위해 임시로 queue_url 스왑
        original_url = self.sqs_client.queue_url
        self.sqs_client.queue_url = self.sqs_client.dlq_url
        
        try:
            attrs = self.sqs_client.get_queue_attributes()
            return {
                'approximate_message_count': int(attrs.get('ApproximateNumberOfMessages', 0)),
                'approximate_not_visible': int(attrs.get('ApproximateNumberOfMessagesNotVisible', 0)),
                'queue_arn': attrs.get('QueueArn', '')
            }
        finally:
            self.sqs_client.queue_url = original_url
    
    def reprocess_message(self, dlq_message: DLQMessage) -> bool:
        """
        DLQ 메시지를 메인 큐로 재전송 (수동 재처리)
        
        Args:
            dlq_message: 재처리할 DLQ 메시지
            
        Returns:
            재전송 성공 여부
        """
        try:
            # 재시도 카운트 리셋하고 메인 큐로 재전송
            reprocessed = dlq_message.original_message.model_copy(update={'retry_count': 0})
            message_id = self.sqs_client.publish(reprocessed)
            
            if message_id:
                logger.info(f"Message reprocessed: schedule_id={reprocessed.schedule_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to reprocess message: {e}")
            return False
