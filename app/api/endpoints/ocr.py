from dependency_injector.wiring import Provide
from fastapi import APIRouter, Depends, HTTPException, status

from app.core.container import Container
from app.core.middleware import inject_ocr
from app.ocr.services.ocr_service import OcrService
from app.ocr.schema.medication_schema import (
    MedicationOCRRequest,
    MedicationOCRResponse
)


router = APIRouter(
    prefix="/ocr",
    tags=["ocr"],
)


@router.get(
    "",
    summary="OCR 서비스 테스트",
    description="OCR 서비스가 정상적으로 작동하는지 테스트합니다. (개발용)"
)
@inject_ocr
def test_ocr_service(
    service: OcrService = Depends(Provide[Container.ocr_service]),
):
    """OCR 서비스 테스트 엔드포인트"""
    return service.test()


@router.post(
    "/validate-medication",
    response_model=MedicationOCRResponse,
    summary="약 정보 OCR 검증",
    description="Luxia OCR 결과를 LLM으로 검증하고 약 정보를 추출합니다."
)
@inject_ocr
async def validate_medication_ocr(
    request: MedicationOCRRequest,
    service: OcrService = Depends(Provide[Container.ocr_service]),
):
    """
    OCR 텍스트를 LLM으로 검증하고 약 정보 추출
    
    **처리 흐름:**
    1. Luxia OCR 원본 텍스트 수신
    2. LLM(GPT)으로 텍스트 분석 및 검증
    3. 약 이름, 용량, 복용 시간, 복용 방법 추출
    4. 신뢰도 점수 계산
    5. 검증 결과 반환
    
    **예시 요청:**
    ```json
    {
      "ocr_text": "타이레놀정 500mg\\n1일 3회\\n식후 30분\\n1회 1정",
      "elderly_user_id": 123
    }
    ```
    
    **예시 응답:**
    ```json
    {
      "success": true,
      "medications": [
        {
          "medication_name": "타이레놀정 500mg",
          "dosage": "1회 1정",
          "times": ["morning", "noon", "evening"],
          "instructions": "식후 30분",
          "confidence": 0.95
        }
      ],
      "llm_analysis": "OCR 텍스트에서 타이레놀정 500mg 약 정보를 추출했습니다...",
      "warnings": []
    }
    ```
    """
    try:
        # async 메서드 호출
        result = await service.validate_medication(
            ocr_text=request.ocr_text,
            elderly_user_id=request.elderly_user_id
        )
        
        # Dict를 MedicationOCRResponse로 변환
        response = MedicationOCRResponse(**result)
        
        # 검증 실패 시 에러 응답
        if not response.success and response.error_message:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=response.error_message
            )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OCR 검증 중 오류 발생: {str(e)}"
        )