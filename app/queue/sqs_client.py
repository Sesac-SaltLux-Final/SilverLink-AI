"""
AWS SQS Client

SQS 큐와의 통신을 담당하는 클라이언트 클래스.
메시지 발행, 수신, 삭제 기능 제공.
"""
import json
import logging
from typing import Optional, List
from datetime import datetime

import boto3
from botocore.exceptions import ClientError

from app.core.config import configs
from app.queue.message_schema import CallRequestMessage

logger = logging.getLogger(__name__)


class SQSClient:
    """
    AWS SQS 클라이언트
    
    메시지 큐 작업을 위한 래퍼 클래스
    """
    
    def __init__(
        self,
        queue_url: str = None,
        dlq_url: str = None,
        region_name: str = None,
        aws_access_key_id: str = None,
        aws_secret_access_key: str = None
    ):
        """
        SQS 클라이언트 초기화
        
        Args:
            queue_url: 메인 SQS 큐 URL
            dlq_url: Dead Letter Queue URL
            region_name: AWS 리전
            aws_access_key_id: AWS Access Key
            aws_secret_access_key: AWS Secret Key
        """
        self.queue_url = queue_url or configs.SQS_QUEUE_URL
        self.dlq_url = dlq_url or configs.SQS_DLQ_URL
        self.region_name = region_name or configs.AWS_REGION
        
        access_key = aws_access_key_id or configs.AWS_ACCESS_KEY_ID
        secret_key = aws_secret_access_key or configs.AWS_SECRET_ACCESS_KEY
        
        try:
            # SQS 클라이언트 생성
            self.client = boto3.client(
                'sqs',
                region_name=self.region_name,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key
            )
        except Exception as e:
            logger.error(f"❌ SQS Client 초기화 실패: {e}")
            raise
    
    def publish(self, message: CallRequestMessage, delay_seconds: int = 0) -> Optional[str]:
        """
        메시지를 SQS 큐에 발행
        
        Args:
            message: 발행할 메시지
            delay_seconds: 메시지 지연 시간 (초, 최대 900)
            
        Returns:
            메시지 ID (성공 시) 또는 None (실패 시)
        """
        try:
            response = self.client.send_message(
                QueueUrl=self.queue_url,
                MessageBody=message.model_dump_json(),
                DelaySeconds=min(delay_seconds, 900),
                MessageAttributes={
                    'MessageType': {
                        'StringValue': 'CallRequest',
                        'DataType': 'String'
                    }
                }
            )
            
            message_id = response.get('MessageId')
            logger.info(f"Message published successfully: {message_id}")
            return message_id
            
        except ClientError as e:
            logger.error(f"Failed to publish message: {e}")
            return None
    
    def receive(
        self,
        max_messages: int = 10,
        wait_time_seconds: int = 20,
        visibility_timeout: int = 60
    ) -> List[dict]:
        """
        SQS 큐에서 메시지 수신 (Long Polling)
        
        Args:
            max_messages: 최대 수신 메시지 수 (1-10)
            wait_time_seconds: Long Polling 대기 시간 (초)
            visibility_timeout: 메시지 가시성 타임아웃 (초)
            
        Returns:
            수신된 메시지 목록
        """
        try:
            response = self.client.receive_message(
                QueueUrl=self.queue_url,
                MaxNumberOfMessages=min(max_messages, 10),
                WaitTimeSeconds=wait_time_seconds,
                VisibilityTimeout=visibility_timeout,
                MessageAttributeNames=['All'],
                AttributeNames=['All']
            )
            
            messages = response.get('Messages', [])
            logger.debug(f"Received {len(messages)} messages from queue")
            return messages
            
        except ClientError as e:
            logger.error(f"Failed to receive messages: {e}")
            return []
    
    def delete(self, receipt_handle: str) -> bool:
        """
        처리 완료된 메시지 삭제
        
        Args:
            receipt_handle: 메시지 수신 핸들
            
        Returns:
            삭제 성공 여부
        """
        try:
            self.client.delete_message(
                QueueUrl=self.queue_url,
                ReceiptHandle=receipt_handle
            )
            logger.debug(f"Message deleted: {receipt_handle[:20]}...")
            return True
            
        except ClientError as e:
            logger.error(f"Failed to delete message: {e}")
            return False
    
    def move_to_dlq(self, message: CallRequestMessage, failure_reason: str) -> Optional[str]:
        """
        실패한 메시지를 DLQ로 이동
        
        Args:
            message: 원본 메시지
            failure_reason: 실패 사유
            
        Returns:
            DLQ 메시지 ID 또는 None
        """
        if not self.dlq_url:
            logger.warning("DLQ URL not configured, skipping DLQ move")
            return None
            
        try:
            dlq_payload = {
                'original_message': message.model_dump(),
                'failure_reason': failure_reason,
                'failed_at': datetime.utcnow().isoformat(),
                'total_attempts': message.retry_count + 1
            }
            
            response = self.client.send_message(
                QueueUrl=self.dlq_url,
                MessageBody=json.dumps(dlq_payload, default=str),
                MessageAttributes={
                    'MessageType': {
                        'StringValue': 'DLQMessage',
                        'DataType': 'String'
                    }
                }
            )
            
            dlq_message_id = response.get('MessageId')
            logger.warning(f"Message moved to DLQ: {dlq_message_id}, reason: {failure_reason}")
            return dlq_message_id
            
        except ClientError as e:
            logger.error(f"Failed to move message to DLQ: {e}")
            return None
    
    def get_queue_attributes(self) -> dict:
        """
        큐 속성 조회 (모니터링용)
        
        Returns:
            큐 속성 딕셔너리
        """
        try:
            response = self.client.get_queue_attributes(
                QueueUrl=self.queue_url,
                AttributeNames=['All']
            )
            return response.get('Attributes', {})
            
        except ClientError as e:
            logger.error(f"Failed to get queue attributes: {e}")
            return {}
