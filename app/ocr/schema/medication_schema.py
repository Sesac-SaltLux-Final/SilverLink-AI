from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import date


class MedicationOCRRequest(BaseModel):
    """OCR 원본 텍스트 요청"""
    ocr_text: str = Field(..., description="Luxia OCR에서 추출된 원본 텍스트")
    elderly_user_id: Optional[int] = Field(None, description="어르신 사용자 ID")


class MedicationInfo(BaseModel):
    """약 정보"""
    medication_name: str = Field(..., description="약 이름")
    dosage: Optional[str] = Field(None, description="용량 (예: 1정, 500mg)")
    times: List[str] = Field(default_factory=list, description="복용 시간 (morning, noon, evening, night)")
    instructions: Optional[str] = Field(None, description="복용 방법 (예: 식후 30분)")
    confidence: float = Field(..., description="신뢰도 (0.0 ~ 1.0)")


class MedicationOCRResponse(BaseModel):
    """OCR 검증 결과"""
    success: bool = Field(..., description="검증 성공 여부")
    medications: List[MedicationInfo] = Field(default_factory=list, description="추출된 약 정보 리스트")
    raw_ocr_text: str = Field(..., description="원본 OCR 텍스트")
    llm_analysis: str = Field(..., description="LLM 분석 결과")
    warnings: List[str] = Field(default_factory=list, description="경고 메시지")
    error_message: Optional[str] = Field(None, description="에러 메시지")


class MedicationScheduleRequest(BaseModel):
    """복약 일정 등록 요청"""
    elderly_user_id: int = Field(..., description="어르신 사용자 ID")
    medication_name: str = Field(..., description="약 이름")
    dosage_text: Optional[str] = Field(None, description="용량")
    times: List[str] = Field(..., description="복용 시간")
    instructions: Optional[str] = Field(None, description="복용 방법")
    start_date: Optional[date] = Field(None, description="시작일")
    end_date: Optional[date] = Field(None, description="종료일")
    reminder: bool = Field(True, description="알림 여부")
