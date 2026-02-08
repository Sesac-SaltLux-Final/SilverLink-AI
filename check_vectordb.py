# -*- coding: utf-8 -*-
"""
벡터DB(Milvus/Zilliz) 데이터 확인 스크립트
실행: python check_vectordb.py
"""
from pymilvus import connections, Collection, utility
from app.core.config import configs

def connect():
    """Milvus 연결"""
    connections.connect(
        alias="default",
        uri=configs.MILVUS_URI,
        token=configs.MILVUS_TOKEN
    )
    print("[OK] Milvus Connected!")

def list_collections():
    """모든 컬렉션 목록"""
    collections = utility.list_collections()
    print(f"\n[Collections]: {collections}")
    return collections

def check_collection(name: str, limit: int = 10):
    """컬렉션 데이터 확인"""
    if not utility.has_collection(name):
        print(f"[ERROR] '{name}' collection not found.")
        return
    
    collection = Collection(name)
    collection.load()
    
    # 컬렉션 정보
    print(f"\n{'='*60}")
    print(f"[Collection]: {name}")
    print(f"  - Total entities: {collection.num_entities}")
    print(f"  - Schema: {[f.name for f in collection.schema.fields]}")
    
    # 샘플 데이터 조회
    if collection.num_entities > 0:
        results = collection.query(
            expr="id >= 0",
            output_fields=["id", "question", "answer"],
            limit=limit
        )
        print(f"\n  [Sample Data] ({min(limit, len(results))} items):")
        for i, item in enumerate(results, 1):
            q = item.get('question', 'N/A')[:80]
            a = item.get('answer', 'N/A')[:80]
            print(f"\n  [{i}] ID: {item.get('id')}")
            print(f"      Q: {q}")
            print(f"      A: {a}")

if __name__ == "__main__":
    connect()
    collections = list_collections()
    
    # FAQ 컬렉션 확인
    check_collection(configs.FAQ_COLLECTION_NAME)
    
    # Inquiry 컬렉션 확인
    check_collection(configs.INQUIRY_COLLECTION_NAME)
    
    print(f"\n{'='*60}")
    print("[DONE]")
