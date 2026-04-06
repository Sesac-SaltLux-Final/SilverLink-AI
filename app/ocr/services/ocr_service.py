import json
import re
from typing import Optional, Dict, Any
from loguru import logger

from app.ocr.repository.ocr_repository import OcrRepository
from app.ocr.services.base_service import BaseService
from app.integration.llm.openai_client import LLM


class OcrService(BaseService):
    def __init__(self, ocr_repository: OcrRepository, llm: LLM):
        self.ocr_repository = ocr_repository
        self.llm = llm
        super().__init__(ocr_repository)
        
    def test(self):
        logger.info("OCR 서비스 테스트")
        return {"status": "ok", "message": "OCR service is running"}
    
    async def validate_medication(self, ocr_text: str, elderly_user_id: Optional[int] = None) -> Dict[str, Any]:
        """
        OCR 텍스트를 LLM으로 분석하여 약 정보를 추출합니다.
        """
        logger.info(f"약 정보 검증 시작 - user_id: {elderly_user_id}")
        logger.debug(f"OCR 텍스트: {ocr_text[:200]}...")
        
        try:
            # LLM 프롬프트 생성
            prompt = self._create_medication_extraction_prompt(ocr_text)
            
            # LLM 호출 (비동기)
            messages = [
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": prompt}
            ]
            
            # 비동기 OpenAI API 호출
            completion = await self.llm.aclient.chat.completions.create(
                model=self.llm.model_version,
                messages=messages
            )
            response = completion.choices[0].message.content
            logger.debug(f"LLM 응답: {response}")
            
            # 응답 파싱
            result = self._parse_llm_response(response, ocr_text)
            logger.info(f"약 정보 추출 완료 - 발견된 약: {len(result['medications'])}개")
            
            return result
            
        except Exception as e:
            logger.error(f"약 정보 검증 실패: {e}")
            return {
                "success": False,
                "medications": [],
                "raw_ocr_text": ocr_text,
                "llm_analysis": "",
                "warnings": [],
                "error_message": str(e)
            }
    
    def _get_system_prompt(self) -> str:
        return """당신은 한국의 약봉투/처방전 OCR 텍스트를 분석하는 전문가입니다.
                    OCR로 추출된 텍스트에서 약 정보를 정확하게 식별하고 구조화된 데이터로 변환해야 합니다.

                    응답은 반드시 다음 JSON 형식으로 해주세요:
        {
            "medications": [
                {
                    "medication_name": "약 이름",
                    "dosage": "용량 (예: 1정, 5mg, 10ml)",
                    "times": ["morning", "noon", "evening", "night"],
                    "instructions": "복용법 (예: 식후 30분)",
                    "confidence": 0.0~1.0 사이의 신뢰도,
                    "category": "약 카테고리"
                }
            ],
            "analysis": "분석 요약 설명 (한국어로 어르신이 이해하기 쉽게)",
            "warnings": ["주의사항 또는 확인이 필요한 사항"]
        }

        복용 시간 매핑:
        - 아침, 조식, morning -> "morning"
        - 점심, 중식, noon -> "noon"
        - 저녁, 석식, evening -> "evening"
        - 취침전, 자기전, night -> "night"

        카테고리 분류 (다음 중 하나를 선택):
        - "혈압약": 고혈압 치료제 (예: 아모디핀, 로사르탄, 노바스크)
        - "당뇨약": 혈당 조절제 (예: 메트포르민, 자디앙스)
        - "감기약": 감기/해열/진해제 (예: 타이레놀, 아세트아미노펜)
        - "위장약": 소화기 치료제 (예: 오메프라졸, 넥시움, 판토프라졸)
        - "진통제": 통증 완화제 (예: 이부프로펜, 아스피린, 세레브렉스)
        - "수면제": 수면 유도제 (예: 졸피뎀, 스틸녹스)
        - "비타민": 영양 보충제 (예: 비타민C, 비타민D)
        - "기타": 위 분류에 해당하지 않는 경우

        신뢰도 기준:
        - 0.9~1.0: 약 이름, 용량, 복용시간이 모두 명확히 식별됨
        - 0.7~0.9: 대부분 식별되었으나 일부 불분명
        - 0.5~0.7: 부분적으로만 식별됨
        - 0.5 미만: 추정값이 많음"""

    def _create_medication_extraction_prompt(self, ocr_text: str) -> str:
        return f"""다음은 약봉투/처방전에서 OCR로 추출된 텍스트입니다. 
                    약 정보를 분석해주세요.

                    === OCR 텍스트 ===
                    {ocr_text}
                    ===================

                    위 텍스트에서 약 정보를 추출하여 JSON 형식으로 응답해주세요.
                    약이 여러 개라면 모두 추출해주세요."""

    def _parse_llm_response(self, response: str, original_text: str) -> Dict[str, Any]:
        """
        LLM 응답을 파싱하여 구조화된 데이터로 변환합니다.
        """
        try:
            # JSON 블록 추출 시도
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # JSON 블록이 없으면 전체 응답을 파싱 시도
                json_str = response
            
            # JSON 파싱
            data = json.loads(json_str)
            
            medications = []
            for med in data.get("medications", []):
                medications.append({
                    "medication_name": med.get("medication_name", "알 수 없는 약"),
                    "dosage": med.get("dosage"),
                    "times": med.get("times", ["morning", "evening"]),
                    "instructions": med.get("instructions"),
                    "confidence": min(max(med.get("confidence", 0.5), 0.0), 1.0),
                    "category": med.get("category", "기타")
                })
            
            return {
                "success": len(medications) > 0,
                "medications": medications,
                "raw_ocr_text": original_text,
                "llm_analysis": data.get("analysis", "약 정보를 분석했습니다."),
                "warnings": data.get("warnings", []),
                "error_message": None
            }
            
        except json.JSONDecodeError as e:
            logger.warning(f"JSON 파싱 실패, 텍스트 기반 추출 시도: {e}")
            return self._fallback_extraction(response, original_text)
    
    def _fallback_extraction(self, response: str, original_text: str) -> Dict[str, Any]:
        """
        JSON 파싱 실패 시 텍스트 기반 폴백 추출
        """
        medications = []
        
        # 간단한 패턴 매칭으로 약 이름 추출
        patterns = [
            r'([가-힣a-zA-Z]+(?:정|캡슐|시럽|액|크림|연고))',
            r'([가-힣]{2,}약)',
            r'([a-zA-Z]+\s*\d+(?:mg|ml|mcg)?)',
        ]
        
        found_meds = set()
        for pattern in patterns:
            matches = re.findall(pattern, original_text)
            for match in matches:
                cleaned = match.strip()
                if len(cleaned) >= 2:
                    found_meds.add(cleaned)
        
        for med_name in list(found_meds)[:5]:  # 최대 5개
            medications.append({
                "medication_name": med_name,
                "dosage": None,
                "times": ["morning", "evening"],
                "instructions": None,
                "confidence": 0.4
            })
        
        return {
            "success": len(medications) > 0,
            "medications": medications,
            "raw_ocr_text": original_text,
            "llm_analysis": "AI 분석이 완료되지 않았습니다. 기본 방식으로 약 이름을 추출했습니다.",
            "warnings": ["자동 분석 결과입니다. 약 정보를 직접 확인해주세요."],
            "error_message": None
        }
