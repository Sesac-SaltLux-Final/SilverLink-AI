-- RAG 평가 지표 저장을 위한 chatbot_logs 테이블 컬럼 추가
-- 실행 전 백업 권장

-- 검색 유사도 점수 (벡터 검색 최고 점수)
ALTER TABLE chatbot_logs ADD COLUMN retrieval_score FLOAT DEFAULT NULL;

-- Q-A 유사도 점수 (질문-답변 코사인 유사도)
ALTER TABLE chatbot_logs ADD COLUMN qa_similarity_score FLOAT DEFAULT NULL;

-- 검색된 컨텍스트 원문
ALTER TABLE chatbot_logs ADD COLUMN retrieved_context TEXT DEFAULT NULL;
