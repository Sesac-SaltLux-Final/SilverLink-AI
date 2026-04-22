"""
병렬 검색 vs 직렬 검색 성능 비교 벤치마크 스크립트

사용법:
    python benchmark_parallel_vs_sequential.py

결과:
    - 병렬 검색 평균 시간
    - 직렬 검색 평균 시간
    - 성능 향상률 (%)
"""

import asyncio
import time
import statistics
from app.chatbot.services.embedding_service import EmbeddingService
from app.chatbot.repository.chatbot_repository import ChatbotRepository


class SearchBenchmark:
    def __init__(self):
        self.chatbot_repository = ChatbotRepository()
        self.embedding_service = EmbeddingService()
        
    async def _search_faq_async(self, embedding: list[float]):
        """FAQ 비동기 검색"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.chatbot_repository.search_faq(embedding, limit=3)
        )

    async def _search_inquiry_async(self, embedding: list[float], guardian_id: int, elderly_id: int):
        """Inquiry 비동기 검색"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.chatbot_repository.search_inquiry(embedding, guardian_id, elderly_id, limit=2)
        )

    async def parallel_search(self, embedding: list[float], guardian_id: int, elderly_id: int):
        """병렬 검색 (현재 방식)"""
        start_time = time.time()
        
        # 두 검색을 동시에 실행
        faq_task = asyncio.create_task(self._search_faq_async(embedding))
        inquiry_task = asyncio.create_task(self._search_inquiry_async(embedding, guardian_id, elderly_id))
        
        faq_results, inquiry_results = await asyncio.gather(faq_task, inquiry_task)
        
        elapsed = time.time() - start_time
        return elapsed, faq_results, inquiry_results

    async def sequential_search(self, embedding: list[float], guardian_id: int, elderly_id: int):
        """직렬 검색 (비교 대상)"""
        start_time = time.time()
        
        # FAQ 먼저 검색
        faq_results = await self._search_faq_async(embedding)
        
        # 그 다음 Inquiry 검색 (순차적으로)
        inquiry_results = await self._search_inquiry_async(embedding, guardian_id, elderly_id)
        
        elapsed = time.time() - start_time
        return elapsed, faq_results, inquiry_results

    async def run_benchmark(self, iterations: int = 10):
        """벤치마크 실행"""
        print("=" * 60)
        print("🚀 병렬 검색 vs 직렬 검색 성능 비교 벤치마크")
        print("=" * 60)
        print(f"테스트 반복 횟수: {iterations}회\n")
        
        # 테스트용 샘플 데이터
        test_queries = [
            "어르신 식사 시간이 언제인가요?",
            "약 복용 방법을 알려주세요",
            "낙상 예방은 어떻게 하나요?",
            "치매 예방에 좋은 활동은?",
            "복지관 이용 방법은?"
        ]
        
        guardian_id = 1
        elderly_id = 1
        
        parallel_times = []
        sequential_times = []
        
        for i in range(iterations):
            query = test_queries[i % len(test_queries)]
            embedding = self.embedding_service.create_embedding(query)
            
            # 병렬 검색 측정
            parallel_time, _, _ = await self.parallel_search(embedding, guardian_id, elderly_id)
            parallel_times.append(parallel_time)
            
            # 직렬 검색 측정
            sequential_time, _, _ = await self.sequential_search(embedding, guardian_id, elderly_id)
            sequential_times.append(sequential_time)
            
            print(f"[{i+1}/{iterations}] 병렬: {parallel_time*1000:.2f}ms | 직렬: {sequential_time*1000:.2f}ms")
        
        # 통계 계산
        avg_parallel = statistics.mean(parallel_times)
        avg_sequential = statistics.mean(sequential_times)
        std_parallel = statistics.stdev(parallel_times) if len(parallel_times) > 1 else 0
        std_sequential = statistics.stdev(sequential_times) if len(sequential_times) > 1 else 0
        
        speedup = ((avg_sequential - avg_parallel) / avg_sequential) * 100
        
        print("\n" + "=" * 60)
        print("📊 벤치마크 결과")
        print("=" * 60)
        print(f"병렬 검색 평균:    {avg_parallel*1000:.2f}ms (± {std_parallel*1000:.2f}ms)")
        print(f"직렬 검색 평균:    {avg_sequential*1000:.2f}ms (± {std_sequential*1000:.2f}ms)")
        print(f"\n⚡ 성능 향상:       {speedup:.1f}% 빠름")
        print(f"💾 시간 절감:       {(avg_sequential - avg_parallel)*1000:.2f}ms")
        print("=" * 60)
        
        # 결과 해석
        print("\n📝 결론:")
        if speedup > 10:
            print(f"✅ 병렬 검색이 {speedup:.1f}% 더 빠릅니다! (유의미한 성능 향상)")
        elif speedup > 0:
            print(f"✅ 병렬 검색이 {speedup:.1f}% 더 빠릅니다.")
        else:
            print(f"⚠️ 이번 테스트에서는 유의미한 차이가 없습니다.")
        
        return {
            "avg_parallel_ms": avg_parallel * 1000,
            "avg_sequential_ms": avg_sequential * 1000,
            "speedup_percent": speedup,
            "time_saved_ms": (avg_sequential - avg_parallel) * 1000
        }


async def main():
    benchmark = SearchBenchmark()
    
    try:
        results = await benchmark.run_benchmark(iterations=10)
        print(f"\n✅ 벤치마크 완료!")
        
    except Exception as e:
        print(f"\n❌ 벤치마크 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
