"""
RAG 평가 스크립트
- 검색 품질 (Retrieval Score)
- Q-A 유사도 (Question-Answer Similarity)
"""
import asyncio
import numpy as np
from langchain_openai import OpenAIEmbeddings
import sys
import os

# 프로젝트 루트 경로 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import configs

# 평가용 테스트 FAQ 데이터 (사용자 제공)
EVALUATION_FAQ_DATA = [
    {
        "question": "상담사가 바뀌면 이전 상담 기록도 새로운 상담사가 볼 수 있나요?",
        "answer": "동일 기관 내에서만 필요한 범위 내에서 이전 상담 기록을 확인할 수 있도록 제한적으로 열람 권한을 부여합니다."
    },
    {
        "question": "서비스 이용 요금은 어떻게 되나요?",
        "answer": "기본 챗봇 기능은 무료로 제공되며, 기관 계약에 따라 일부 프리미엄 기능은 별도 요금 정책이 적용될 수 있습니다."
    },
    {
        "question": "콜봇 전화를 받지 못하면 어떻게 되나요?",
        "answer": "일정 시간 동안 연결이 되지 않으면 콜봇이 다시 한 번 전화를 시도하고, 계속 응답이 없을 경우 보호자 앱에 '통화 실패' 알림이 전송됩니다."
    },
    {
        "question": "비밀번호를 잊어버렸는데 어떻게 해야 하나요?",
        "answer": "로그인 화면의 '비밀번호 찾기'를 눌러 인증 절차를 진행하면 새 비밀번호를 설정할 수 있습니다."
    }
]


def calculate_cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """두 벡터 간 코사인 유사도 계산"""
    v1, v2 = np.array(vec1), np.array(vec2)
    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


def evaluate_qa_similarity():
    """FAQ 데이터 기반 Q-A 유사도 평가"""
    
    print("=" * 60)
    print("🎯 RAG 평가: Q-A 유사도 테스트")
    print("=" * 60)
    
    # 임베딩 서비스 초기화
    embeddings = OpenAIEmbeddings(
        model=configs.EMBEDDING_MODEL,
        api_key=configs.OPENAI_API_KEY
    )
    
    results = []
    
    for idx, faq in enumerate(EVALUATION_FAQ_DATA):
        question = faq["question"]
        ground_truth_answer = faq["answer"]
        
        # 질문과 정답 임베딩 생성
        q_embedding = embeddings.embed_query(question)
        a_embedding = embeddings.embed_query(ground_truth_answer)
        
        # Q-A 유사도 계산 (질문 ↔ 정답)
        qa_similarity = calculate_cosine_similarity(q_embedding, a_embedding)
        
        results.append({
            "id": idx + 1,
            "question": question[:40] + "..." if len(question) > 40 else question,
            "qa_similarity": qa_similarity
        })
        
        print(f"\n[{idx + 1}] Q: {question[:50]}...")
        print(f"    A: {ground_truth_answer[:50]}...")
        print(f"    📊 Q-A 유사도: {qa_similarity:.4f}")
    
    # 통계 출력
    avg_similarity = np.mean([r["qa_similarity"] for r in results])
    min_similarity = np.min([r["qa_similarity"] for r in results])
    max_similarity = np.max([r["qa_similarity"] for r in results])
    
    print("\n" + "=" * 60)
    print("📈 평가 결과 요약")
    print("=" * 60)
    print(f"  • 평균 Q-A 유사도: {avg_similarity:.4f}")
    print(f"  • 최소 Q-A 유사도: {min_similarity:.4f}")
    print(f"  • 최대 Q-A 유사도: {max_similarity:.4f}")
    print(f"  • 테스트 케이스 수: {len(results)}")
    
    # 품질 등급 판정
    if avg_similarity >= 0.80:
        grade = "A (우수)"
    elif avg_similarity >= 0.70:
        grade = "B (양호)"
    elif avg_similarity >= 0.60:
        grade = "C (보통)"
    else:
        grade = "D (개선 필요)"
    
    print(f"  • 종합 등급: {grade}")
    print("=" * 60)
    
    return results


async def evaluate_chatbot_responses():
    """실제 챗봇 API를 호출하여 응답 평가"""
    
    print("\n" + "=" * 60)
    print("🤖 RAG 평가: 실제 챗봇 응답 테스트")
    print("=" * 60)
    
    from app.chatbot.services.chatbot_service import ChatbotService
    from app.chatbot.repository.chatbot_repository import ChatbotRepository
    
    # 서비스 초기화
    repository = ChatbotRepository()
    service = ChatbotService(repository)
    
    embeddings = OpenAIEmbeddings(
        model=configs.EMBEDDING_MODEL,
        api_key=configs.OPENAI_API_KEY
    )
    
    results = []
    
    for idx, faq in enumerate(EVALUATION_FAQ_DATA):
        question = faq["question"]
        ground_truth_answer = faq["answer"]
        
        print(f"\n[{idx + 1}] 질문: {question[:50]}...")
        
        try:
            # 챗봇 응답 생성
            response = await service.process_chat(
                message=question,
                thread_id=f"eval_test_{idx}",
                guardian_id=1,
                elderly_id=1
            )
            
            bot_answer = response["answer"]
            retrieval_score = response.get("confidence", 0.0)
            
            # 임베딩 생성
            q_embedding = embeddings.embed_query(question)
            bot_a_embedding = embeddings.embed_query(bot_answer)
            gt_a_embedding = embeddings.embed_query(ground_truth_answer)
            
            # 유사도 계산
            qa_similarity = calculate_cosine_similarity(q_embedding, bot_a_embedding)
            answer_accuracy = calculate_cosine_similarity(bot_a_embedding, gt_a_embedding)
            
            results.append({
                "id": idx + 1,
                "question": question,
                "ground_truth": ground_truth_answer,
                "bot_answer": bot_answer,
                "retrieval_score": retrieval_score,
                "qa_similarity": qa_similarity,
                "answer_accuracy": answer_accuracy
            })
            
            print(f"    📝 챗봇 응답: {bot_answer[:60]}...")
            print(f"    📊 검색 점수: {retrieval_score:.4f}")
            print(f"    📊 Q-A 유사도: {qa_similarity:.4f}")
            print(f"    📊 정답 일치도: {answer_accuracy:.4f}")
            
        except Exception as e:
            print(f"    ❌ 오류 발생: {e}")
            continue
    
    if results:
        # 통계 출력
        avg_retrieval = np.mean([r["retrieval_score"] for r in results])
        avg_qa_sim = np.mean([r["qa_similarity"] for r in results])
        avg_accuracy = np.mean([r["answer_accuracy"] for r in results])
        
        print("\n" + "=" * 60)
        print("📈 챗봇 응답 평가 결과 요약")
        print("=" * 60)
        print(f"  • 평균 검색 점수: {avg_retrieval:.4f}")
        print(f"  • 평균 Q-A 유사도: {avg_qa_sim:.4f}")
        print(f"  • 평균 정답 일치도: {avg_accuracy:.4f}")
        print(f"  • 테스트 케이스 수: {len(results)}")
        print("=" * 60)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG 평가 스크립트")
    parser.add_argument("--mode", choices=["qa", "chatbot", "all"], default="qa",
                       help="평가 모드 선택: qa(Q-A 유사도만), chatbot(실제 챗봇), all(모두)")
    args = parser.parse_args()
    
    if args.mode in ["qa", "all"]:
        evaluate_qa_similarity()
    
    if args.mode in ["chatbot", "all"]:
        asyncio.run(evaluate_chatbot_responses())
