from app.ocr.repository.ocr_repository import OcrRepository
from app.ocr.services.base_service import BaseService
from app.ocr.services.medication_validator import MedicationValidator
from app.ocr.schema.medication_schema import (
    MedicationOCRRequest,
    MedicationOCRResponse,
    MedicationInfo
)
from app.integration.llm.openai_client import LLM
from typing import Dict, Any


class OcrService(BaseService):
    def __init__(self, ocr_repository: OcrRepository, llm: LLM):
        self.ocr_repository = ocr_repository
        self.llm = llm
        self.validator = MedicationValidator(llm)
        super().__init__(ocr_repository)
        
    def test(self):
        print('OCR Service Test')
        return {"message": "OCR Service is working"}
    
    def validate_medication_ocr(self, request: MedicationOCRRequest) -> MedicationOCRResponse:
        """
        OCR 텍스트를 LLM으로 검증하고 약 정보 추출
        
        Args:
            request: OCR 텍스트 및 어르신 ID
            
        Returns:
            검증 결과 및 추출된 약 정보
        """
        try:
            # LLM 검증 및 추출
            result = self.validator.validate_and_extract(request.ocr_text)
            
            # Pydantic 모델로 변환
            response = MedicationOCRResponse(
                success=result["success"],
                medications=result["medications"],
                raw_ocr_text=result["raw_ocr_text"],
                llm_analysis=result["llm_analysis"],
                warnings=result["warnings"],
                error_message=result.get("error_message")
            )
            
            return response
            
        except Exception as e:
            return MedicationOCRResponse(
                success=False,
                medications=[],
                raw_ocr_text=request.ocr_text,
                llm_analysis="",
                warnings=["서비스 처리 중 오류 발생"],
                error_message=str(e)
            )
    
    def extract_medications_from_text(self, ocr_text: str) -> Dict[str, Any]:
        """
        OCR 텍스트에서 약 정보만 추출 (간단한 버전)
        
        Args:
            ocr_text: OCR 원본 텍스트
            
        Returns:
            추출된 약 정보
        """
        result = self.validator.validate_and_extract(ocr_text)
        return result