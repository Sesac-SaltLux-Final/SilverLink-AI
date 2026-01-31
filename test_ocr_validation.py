"""
OCR LLM 검증 테스트 스크립트
"""
import os
import sys
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.integration.llm.openai_client import LLM
from app.ocr.services.medication_validator import MedicationValidator


def test_medication_validation():
    """약 정보 검증 테스트"""
    
    # LLM 초기화
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        return
    
    llm = LLM(model_version="gpt-4o-mini", api_key=api_key)
    validator = MedicationValidator(llm)
    
    # 테스트 케이스 1: 단일 약
    print("\n" + "="*60)
    print("테스트 1: 단일 약 정보")
    print("="*60)
    
    ocr_text_1 = """타이레놀정 500mg
1일 3회
식후 30분
1회 1정"""
    
    result_1 = validator.validate_and_extract(ocr_text_1)
    print_result(result_1)
    
    # 테스트 케이스 2: 여러 약
    print("\n" + "="*60)
    print("테스트 2: 여러 약 정보")
    print("="*60)
    
    ocr_text_2 = """1. 타이레놀정 500mg
   1일 3회, 식후 30분, 1회 1정

2. 아스피린 100mg
   1일 1회, 아침 식후, 1회 1정

3. 비타민D 1000IU
   1일 1회, 아침, 1회 1캡슐"""
    
    result_2 = validator.validate_and_extract(ocr_text_2)
    print_result(result_2)
    
    # 테스트 케이스 3: 복잡한 OCR 텍스트 (메타데이터 포함)
    print("\n" + "="*60)
    print("테스트 3: 복잡한 OCR 텍스트")
    print("="*60)
    
    ocr_text_3 = """환자정보: 홍길동(만65세/남)
교부번호: 2024-001234
병원정보: 서울대학교병원
조제 약사: 김약사
처방 의사: 이의사
처방 일자: 2024-01-30

[처방 약품]
타이레놀정 500mg
1일 3회 (아침, 점심, 저녁)
식후 30분
1회 1정

주의사항: 공복 복용 금지"""
    
    result_3 = validator.validate_and_extract(ocr_text_3)
    print_result(result_3)


def print_result(result):
    """결과 출력"""
    print(f"\n✅ 성공: {result['success']}")
    print(f"📝 원본 텍스트:\n{result['raw_ocr_text'][:100]}...")
    print(f"\n🤖 LLM 분석:\n{result['llm_analysis']}")
    
    if result['medications']:
        print(f"\n💊 추출된 약 정보 ({len(result['medications'])}개):")
        for i, med in enumerate(result['medications'], 1):
            print(f"\n  {i}. {med.medication_name}")
            print(f"     - 용량: {med.dosage or 'N/A'}")
            print(f"     - 복용 시간: {', '.join(med.times)}")
            print(f"     - 복용 방법: {med.instructions or 'N/A'}")
            print(f"     - 신뢰도: {med.confidence:.2f}")
    else:
        print("\n⚠️ 추출된 약 정보 없음")
    
    if result['warnings']:
        print(f"\n⚠️ 경고:")
        for warning in result['warnings']:
            print(f"  - {warning}")
    
    if result.get('error_message'):
        print(f"\n❌ 에러: {result['error_message']}")


if __name__ == "__main__":
    print("🧪 OCR LLM 검증 테스트 시작")
    test_medication_validation()
    print("\n✅ 테스트 완료")
