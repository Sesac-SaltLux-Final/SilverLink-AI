import json
import re
from typing import List, Dict, Any
from app.integration.llm.openai_client import LLM
from app.ocr.schema.medication_schema import MedicationInfo


class MedicationValidator:
    """LLM을 활용한 약 정보 검증 및 추출"""
    
    def __init__(self, llm: LLM):
        self.llm = llm
    
    def validate_and_extract(self, ocr_text: str) -> Dict[str, Any]:
        """
        OCR 텍스트를 LLM으로 검증하고 약 정보 추출
        
        Args:
            ocr_text: Luxia OCR에서 추출된 원본 텍스트
            
        Returns:
            검증 결과 및 추출된 약 정보
        """
        try:
            # LLM 프롬프트 생성
            messages = self._create_validation_prompt(ocr_text)
            
            # LLM 호출
            llm_response = self.llm.gpt(messages)
            
            # LLM 응답 파싱
            result = self._parse_llm_response(llm_response, ocr_text)
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "medications": [],
                "raw_ocr_text": ocr_text,
                "llm_analysis": "",
                "warnings": [],
                "error_message": f"LLM 검증 중 오류 발생: {str(e)}"
            }
    
    def _create_validation_prompt(self, ocr_text: str) -> List[Dict[str, str]]:
        """LLM 검증 프롬프트 생성"""
        
        system_prompt = """당신은 약봉투 OCR 텍스트를 분석하는 전문가입니다.
주어진 OCR 텍스트에서 약 정보를 정확하게 추출하고 검증하세요.

**추출해야 할 정보:**
1. 약 이름 (medication_name): 정확한 약품명
2. 용량 (dosage): 1정, 500mg 등
3. 복용 시간 (times): morning(아침), noon(점심), evening(저녁), night(취침전)
4. 복용 방법 (instructions): 식전, 식후 30분 등
5. 신뢰도 (confidence): 0.0 ~ 1.0 (추출 정보의 확실성)

**규칙:**
- 약 이름은 반드시 포함되어야 함
- 복용 시간은 ["morning", "noon", "evening", "night"] 중 선택
- "1일 3회" → ["morning", "noon", "evening"]
- "1일 2회" → ["morning", "evening"]
- "1일 1회" → ["morning"]
- 불확실한 정보는 신뢰도를 낮게 설정
- 여러 약이 있으면 모두 추출 (최대 5개)

**응답 형식 (JSON):**
```json
{
  "medications": [
    {
      "medication_name": "타이레놀정 500mg",
      "dosage": "1회 1정",
      "times": ["morning", "noon", "evening"],
      "instructions": "식후 30분",
      "confidence": 0.95
    }
  ],
  "analysis": "OCR 텍스트 분석 결과 설명",
  "warnings": ["경고 메시지 (있는 경우)"]
}
```"""

        user_prompt = f"""다음 OCR 텍스트에서 약 정보를 추출하고 검증하세요:

```
{ocr_text}
```

위 텍스트를 분석하여 JSON 형식으로 응답해주세요."""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    def _parse_llm_response(self, llm_response: str, ocr_text: str) -> Dict[str, Any]:
        """LLM 응답 파싱"""
        try:
            # JSON 추출 (마크다운 코드 블록 제거)
            json_match = re.search(r'```json\s*(.*?)\s*```', llm_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # 코드 블록 없이 JSON만 있는 경우
                json_str = llm_response.strip()
            
            # JSON 파싱
            parsed = json.loads(json_str)
            
            # MedicationInfo 객체로 변환
            medications = []
            for med_data in parsed.get("medications", []):
                # 필수 필드 검증
                if not med_data.get("medication_name"):
                    continue
                
                # times 필드 검증 및 정제
                times = med_data.get("times", ["morning"])
                valid_times = ["morning", "noon", "evening", "night"]
                times = [t for t in times if t in valid_times]
                if not times:
                    times = ["morning"]  # 기본값
                
                medication = MedicationInfo(
                    medication_name=med_data["medication_name"],
                    dosage=med_data.get("dosage"),
                    times=times,
                    instructions=med_data.get("instructions"),
                    confidence=float(med_data.get("confidence", 0.8))
                )
                medications.append(medication)
            
            return {
                "success": True,
                "medications": medications,
                "raw_ocr_text": ocr_text,
                "llm_analysis": parsed.get("analysis", ""),
                "warnings": parsed.get("warnings", []),
                "error_message": None
            }
            
        except json.JSONDecodeError as e:
            # JSON 파싱 실패 시 폴백: 기본 추출 로직
            return self._fallback_extraction(ocr_text, llm_response)
        except Exception as e:
            return {
                "success": False,
                "medications": [],
                "raw_ocr_text": ocr_text,
                "llm_analysis": llm_response,
                "warnings": ["LLM 응답 파싱 실패"],
                "error_message": f"응답 파싱 오류: {str(e)}"
            }
    
    def _fallback_extraction(self, ocr_text: str, llm_response: str) -> Dict[str, Any]:
        """LLM 응답 파싱 실패 시 폴백 로직"""
        
        # 간단한 패턴 매칭으로 약 이름 추출
        lines = ocr_text.split('\n')
        medication_name = "인식된 약"
        
        for line in lines:
            line = line.strip()
            # 약 이름으로 추정되는 라인 (한글+숫자+mg/정 포함)
            if re.search(r'[가-힣]+.*?(mg|정|캡슐|포)', line, re.IGNORECASE):
                medication_name = line
                break
        
        # 복용 시간 추출
        times = ["morning"]
        if "1일 3회" in ocr_text or "하루 3회" in ocr_text:
            times = ["morning", "noon", "evening"]
        elif "1일 2회" in ocr_text or "하루 2회" in ocr_text:
            times = ["morning", "evening"]
        
        medication = MedicationInfo(
            medication_name=medication_name,
            dosage=None,
            times=times,
            instructions=None,
            confidence=0.5  # 낮은 신뢰도
        )
        
        return {
            "success": True,
            "medications": [medication],
            "raw_ocr_text": ocr_text,
            "llm_analysis": llm_response,
            "warnings": ["LLM 응답 파싱 실패로 기본 추출 로직 사용"],
            "error_message": None
        }
