"""
schemas.py
--------------------------------
FastAPI Request / Response 스키마 정의.

Advanced RAG API는 입력이 question 하나뿐이지만, 교육용 예제이므로
Response에는 LangGraph 내부 상태(route, search_strategy, category, sources,
metrics)를 함께 반환해 학생들이 파이프라인 동작을 눈으로 확인할 수 있게 합니다.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RAGRequest(BaseModel):
    """POST /rag/query 요청 Body"""

    question: str = Field(
        ...,
        description="사용자 질문",
        json_schema_extra={"example": "예비창업자나 스타트업이 할수 지원할 수 있는 사업을 알려주세요"},
    )


class RAGResponse(BaseModel):
    """POST /rag/query 응답 Body"""

    question: str = Field(..., description="원본 질문")
    route: str = Field(..., description="analyze_query가 판단한 라우팅 결과 (rag / general)")
    search_strategy: str = Field(..., description="선택된 검색 전략 (basic / multi_query / hyde / hybrid / none)")
    category: str = Field(..., description="질문 카테고리 (travel_product / hotel / flight / schedule_change / cancellation / general)")
    answer: str = Field(..., description="LangGraph가 생성한 최종 답변")
    sources: list[str] = Field(default_factory=list, description="Reranking을 통과한 최종 참고 문서 출처 파일명 목록")
    contexts: list[str] = Field(
        default_factory=list,
        description=(
            "Reranking을 통과한 최종 참고 문서의 원문(passage) 목록. "
            "sources는 파일명만 담지만, RAGAS 같은 평가 도구는 실제 검색된 "
            "텍스트가 필요하므로 별도 필드로 제공합니다."
        ),
    )
    metrics: dict[str, Any] = Field(default_factory=dict, description="각 단계별 실행 시간 등 지표")

    class Config:
        json_schema_extra = {
            "example": {
                "question": "예비창업자나 스타트업이 할수 지원할 수 있는 사업을 알려주세요",
                "route": "rag",
                "search_strategy": "multi_query",
                "category": "general",
                "answer": "제공된 문서에서 예비창업자나 스타트업이 지원받을 수 있는 사업은 다음과 같습니다 ...",
                "sources": ["./rag_data\\(공고문)_2026년도_중앙부처_및_지자체_창업지원사업_통합공고문(제2025-648호,_2025.12.19.).pdf"],
                "contexts": ["16 -\n연\n번 사업명 사업개요 지원내용 지원대상 예산\n(억원)\n사업\n공고일 소관 부처 전문(주관)\n기관 비고\n12 "],
                "metrics": {
                    "analyze_time": 5.409430742263794,
                    "multi_query_time": 1.7617526054382324,
                    "retrieval_time": 2.295433282852173,
                    "num_search_queries": 4,
                    "num_raw_documents": 12,
                    "post_processing_time": 0,
                    "num_unique_documents": 12,
                    "num_processed_documents": 3,
                    "reranking_time": 2.9948549270629883,
                    "num_reranked_documents": 3,
                    "compression_time": 37.063759088516235,
                    "context_len_before": 1437,
                    "context_len_after": 1379,
                    "generation_time": 13.726775407791138,
                    "total_time": 63.25200605392456
                },
            }
        }
