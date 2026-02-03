"""
SQS Message Schema Definitions

Pydantic models for AWS SQS message serialization/deserialization.
"""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field
from enum import Enum


class CallStatus(str, Enum):
    """통화 상태 enum"""
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class CallRequestMessage(BaseModel):
    """
    SQS로 전송되는 통화 요청 메시지 스키마
    
    Spring Boot BE에서 발행하는 메시지 형식과 일치해야 함
    """
    message_id: str = Field(..., description="고유 메시지 ID")
    # schedule_id: int = Field(..., description="통화 스케줄 ID")
    elderly_id: int = Field(..., description="어르신 ID")
    elderly_name: str = Field(..., description="어르신 이름")
    phone_number: str = Field(..., description="전화번호 (E.164 형식: +821012345678)")
    # scheduled_time: datetime = Field(..., description="예약된 통화 시간")
    retry_count: int = Field(default=0, description="재시도 횟수")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="메시지 생성 시간")


class CallResultMessage(BaseModel):
    """
    통화 결과 메시지 스키마
    
    워커가 처리 결과를 기록하거나 BE에 콜백할 때 사용
    """
    message_id: str = Field(..., description="원본 요청 메시지 ID")
    # schedule_id: int = Field(..., description="통화 스케줄 ID")
    call_sid: Optional[str] = Field(None, description="Twilio Call SID")
    status: CallStatus = Field(..., description="통화 상태")
    duration_seconds: Optional[int] = Field(None, description="통화 시간 (초)")
    error_message: Optional[str] = Field(None, description="에러 메시지 (실패 시)")
    processed_at: datetime = Field(default_factory=datetime.utcnow, description="처리 완료 시간")



class DLQMessage(BaseModel):
    """
    Dead Letter Queue 메시지 스키마
    
    최대 재시도 후 DLQ로 이동된 메시지
    """
    original_message: CallRequestMessage = Field(..., description="원본 요청 메시지")
    failure_reason: str = Field(..., description="실패 사유")
    failed_at: datetime = Field(default_factory=datetime.utcnow, description="최종 실패 시간")
    total_attempts: int = Field(..., description="총 시도 횟수")
