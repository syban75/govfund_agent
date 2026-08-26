"""
main.py
--------------------------------
FastAPI Backend - Advanced RAG를 웹 API로 서비스합니다.

이번 업데이트에서 SSE(Server-Sent Events) 엔드포인트를 추가했습니다.
LangGraph는 analyze_query -> routing -> retrieval -> post_processing ->
reranking -> context_compression -> generate_answer 순으로 여러 노드를
거치기 때문에, 전체가 끝날 때까지 기다리는 대신 노드가 끝날 때마다
중간 진행 상황을 실시간으로 흘려보내는 것이 학습 목적에 더 적합합니다.

핵심 흐름 (SSE):
    POST /rag/query/stream
        -> question
        -> new_state(question)
        -> rag_graph.stream(state, stream_mode="updates")
        -> 노드가 끝날 때마다 "node_update" 이벤트 전송
        -> 그래프 종료 후 "done" 이벤트로 최종 결과 전송
        -> 예외 발생 시 "error" 이벤트 전송

기존 POST /rag/query(동기/단건 응답)는 Swagger에서 간단히 테스트하거나
스트리밍이 필요 없는 클라이언트를 위해 그대로 유지합니다.

주의:
- 이 서버는 Vector DB(chroma_travel_db)를 새로 만들지 않습니다.
  반드시 먼저 `python AdvancedRAG_All_BuildDB.py`를 실행해 DB를 만들어 두세요.
- advanced_rag_agent.py 가 import 되는 시점에 LLM / Embedding / Reranker /
  LangGraph가 모두 초기화되므로, 서버 기동에 다소 시간이 걸릴 수 있습니다.

실행:
    uvicorn main:app --reload

Swagger UI:
    http://127.0.0.1:8014/docs
"""

from __future__ import annotations

import json
from typing import Any, Iterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from govfund_agent import new_state, rag_graph
from schemas import RAGRequest, RAGResponse

app = FastAPI(
    title="Advanced RAG API",
    description="LangGraph 기반 정부 지원 사업 Advanced RAG를 서비스하는 FastAPI 백엔드",
    version="1.0.0",
)

# 노드 이름 -> 화면에 보여줄 한글 라벨
NODE_LABELS: dict[str, str] = {
    "analyze_query": "질문 분석",
    "basic_query": "검색 전략 결정: Basic",
    "multi_query": "Multi-Query 생성",
    "hyde": "HyDE 가상 문서 생성",
    "general_answer": "일반 답변 생성",
    "retrieval": "Vector DB 검색",
    "post_processing": "Post-Processing",
    "reranking": "Reranking",
    "context_compression": "Context Compression",
    "generate_answer": "최종 답변 생성",
    "show_result": "결과 요약",
}


def _sse_event(event: str, data: dict[str, Any]) -> str:
    """SSE 규격(text/event-stream)에 맞춰 한 개의 이벤트 문자열을 만듭니다."""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def _build_final_payload(final_state: dict[str, Any], question: str) -> dict[str, Any]:
    reranked_documents = final_state.get("reranked_documents") or []
    sources = sorted({d["source"] for d in reranked_documents})
    # RAGAS 같은 평가 도구는 파일명(sources)이 아니라 실제 검색된 원문이 필요하므로 함께 반환합니다.
    contexts = [d["content"] for d in reranked_documents]

    return {
        "question": final_state.get("question", question),
        "route": final_state.get("route", ""),
        "search_strategy": final_state.get("search_strategy", ""),
        "category": final_state.get("category", ""),
        "answer": final_state.get("answer", ""),
        "sources": sources,
        "contexts": contexts,
        "metrics": final_state.get("metrics", {}),
    }


def event_generator(question: str) -> Iterator[str]:
    """rag_graph를 스트리밍 실행하며 SSE 이벤트 문자열을 순서대로 yield 합니다.

    엔드포인트 함수 밖으로 분리했기 때문에 `question`을 클로저로 캡처하지 않고
    인자로 명시적으로 받습니다. 덕분에:
      - FastAPI 없이도 `list(event_generator("질문"))` 형태로 단독 테스트 가능
      - 다른 엔드포인트나 배치 스크립트에서도 재사용 가능
      - 함수 시그니처만 보고 무엇이 입력/출력인지 바로 파악 가능
    """
    state = new_state(question)
    # LangGraph는 노드가 반환한 "변경분(update)"만 넘겨주므로,
    # 지금까지의 최종 상태를 직접 누적해서 관리합니다.
    final_state: dict[str, Any] = dict(state)

    try:
        for chunk in rag_graph.stream(state, stream_mode="updates"):
            # chunk 예: {"analyze_query": {"route": "rag", "search_strategy": "basic", ...}}
            for node_name, node_output in chunk.items():
                final_state.update(node_output)

                logs = node_output.get("logs")
                latest_log = logs[-1] if logs else None

                payload = {
                    "node": node_name,
                    "label": NODE_LABELS.get(node_name, node_name),
                    "message": latest_log,
                    "route": final_state.get("route", ""),
                    "search_strategy": final_state.get("search_strategy", ""),
                    "category": final_state.get("category", ""),
                }
                yield _sse_event("node_update", payload)

        yield _sse_event("done", _build_final_payload(final_state, question))

    except Exception as e:
        # 예: Ollama 서버 크래시(CUDA 에러) 등 파이프라인 실행 중 예외를
        # 500 에러 대신 SSE error 이벤트로 클라이언트에 전달합니다.
        yield _sse_event("error", {"message": str(e)})


# ---------------------------------------------------------------------------
# ① 서버 확인
# ---------------------------------------------------------------------------
@app.get("/")
def read_root():
    return {"message": "Advanced RAG API is running"}


# ---------------------------------------------------------------------------
# ② 상태 확인
# ---------------------------------------------------------------------------
@app.get("/health")
def health_check():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# ③ Advanced RAG 실행 (동기 / 단건 응답) - Swagger 테스트용으로 유지
# ---------------------------------------------------------------------------
@app.post("/rag/query", response_model=RAGResponse)
def query_rag(request: RAGRequest):
    question = (request.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="question은 비어 있을 수 없습니다.")

    state = new_state(question)
    result = rag_graph.invoke(state)

    return RAGResponse(**_build_final_payload(result, question))


# ---------------------------------------------------------------------------
# ④ Advanced RAG 실행 (SSE 스트리밍) - Streamlit이 사용하는 엔드포인트
# ---------------------------------------------------------------------------
@app.post("/rag/query/stream")
def query_rag_stream(request: RAGRequest):
    question = (request.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="question은 비어 있을 수 없습니다.")

    return StreamingResponse(
        event_generator(question),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # 프록시(nginx 등) 버퍼링 방지
        },
    )
