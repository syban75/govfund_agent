"""
govfund_agent.py
--------------------------------
LangGraph 기반 Advanced RAG(질문 분석 / Multi-Query / HyDE / 검색 /
Post-Processing / Reranking / Context Compression / 답변 생성)를
그대로 유지하되, main.py(FastAPI)에서 import 해서 쓸 수 있도록
그래프를 모듈 최초 로딩 시 한 번만 컴파일합니다.

FastAPI의 `app = FastAPI()` 와 이름이 겹치지 않도록, 여기서는
컴파일된 그래프 객체 이름을 `rag_graph` 로 둡니다.

사전 준비:
- govfund_BuildDB.py 를 먼저 실행하여 chroma_travel_db 생성
  (이 모듈은 DB를 다시 만들지 않고 검색만 수행합니다)
- Ollama 로컬 서버 실행 (ollama serve) 및 모델 다운로드:
    ollama pull gemma4:e4b   (또는 사용할 LLM 모델)
    ollama pull bge-m3       (임베딩 모델, BuildDB와 동일해야 함)
- pip install -r requirements.txt
- Reranking 모델(Dongjin-kr/ko-reranker)은 최초 실행 시 Hugging Face에서 자동 다운로드됩니다.
"""

from __future__ import annotations

import json
import re
import time
from typing import TypedDict, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph import StateGraph, START, END

# ✅ 문제가 많던 langchain.retrievers 대신, AI 모델 라이브러리를 직접 사용합니다.
from sentence_transformers import CrossEncoder

try:
    from langchain_chroma import Chroma
except ImportError:  # langchain-chroma 미설치 환경 대비
    from langchain_community.vectorstores import Chroma

# ---------------------------------------------------------------------------
# 설정값 (govfund_BuildDB.py 와 동일하게 맞춰야 합니다)
# ---------------------------------------------------------------------------

DB_DIR = "../chroma_govfund_db"
COLLECTION_NAME = "govfund_guide"
EMBEDDING_MODEL = "bge-m3"        # 사용자 환경에 맞게 변경 (ollama pull bge-m3 필요)
OLLAMA_MODEL = "qwen2.5:14b" #"gemma4:e4b"      # 사용자 환경에 맞게 변경
RERANK_MODEL_NAME = "Dongjin-kr/ko-reranker"

TOP_K_PER_QUERY = 3          # Query 1개당 검색할 문서 수
POST_PROCESS_LIMIT = 8       # Post-Processing 후 Reranking으로 넘길 후보 문서 수
RERANK_TOP_N = 5             # Reranking 이후 최종 유지 문서 수
SCORE_THRESHOLD = 0.5        # relevance_score 최소 기준 (이 값 미만 제거)

CATEGORY_MAP = {
    "(공고문)_2026년도_중앙부처_및_지자체_창업지원사업_통합공고문(제2025-648호,_2025.12.19.).pdf": "notice",
}

embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url="http://10.8.0.1:11434")

try:
    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0,base_url="http://10.8.0.1:11434")
except Exception as e:
    print(f"⚠️ LLM 초기화 실패: {e}")

try:
    # device="cpu": GPU를 Ollama(임베딩/LLM)와 동시에 점유하면
    # CUDA 컨텍스트 충돌로 ollama의 llama-server가 크래시할 수 있어 CPU로 고정
    reranker_model = CrossEncoder(RERANK_MODEL_NAME, device="cpu")
except Exception as e:
    reranker_model = None
    print(f"⚠️ Reranker 초기화 실패: {e}")

vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=DB_DIR,
)


# ---------------------------------------------------------------------------
# State 설계
# ---------------------------------------------------------------------------
class AdvancedRAGState(TypedDict, total=False):
    question: str
    route: str
    search_strategy: str
    category: str
    multi_queries: list[str]
    hypothetical_document: str
    search_queries: list[str]
    raw_documents: list[dict]
    processed_documents: list[dict]
    reranked_documents: list[dict]
    compressed_context: str
    answer: str
    logs: list[str]
    metrics: dict

def new_state(question: str) -> AdvancedRAGState:
    return {
        "question": question,
        "route": "",
        "search_strategy": "",
        "category": "",
        "multi_queries": [],
        "hypothetical_document": "",
        "search_queries": [],
        "raw_documents": [],
        "processed_documents": [],
        "reranked_documents": [],
        "compressed_context": "",
        "answer": "",
        "logs": [],
        "metrics": {},
    }



# ---------------------------------------------------------------------------
# 노드 1. analyze_query_node : 질문 분석 및 전략/카테고리 판단
# ---------------------------------------------------------------------------
ANALYZE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "당신은 정부 지원 사업 챗봇의 질문 분석기입니다. "
            "아래 JSON 형식으로만 답변하고, 다른 설명은 절대 출력하지 마세요.\n"
            "{{\n"
            '  "route": "rag" 또는 "general",\n'
            '  "strategy": "basic" 또는 "multi_query" 또는 "hyde" 또는 "hybrid" 또는 "none",\n'
            '  "category": "notice" ,\n '
            '  "schedule_change" 또는 "cancellation" 또는 "general"\n'
            "}}\n\n"
            "판단 기준:\n"
            "- 정부 지원 사업과 관련된 질문이면 route는 rag\n"
            "- 문서와 무관한 일반 지식 질문이면 route는 general, strategy는 none\n"
            "- 사실 하나를 정확히 찾는 질문이면 strategy는 basic\n"
            "- 표현이 다양하거나 비교가 필요한 질문이면 strategy는 multi_query\n"
            "- 질문이 짧거나 검색 키워드가 부족하면 strategy는 hyde\n"
            "- 여러 조건이 복합적으로 포함된 질문이면 strategy는 hybrid",
        ),
        ("human", "질문: {question}"),
    ]
)


def analyze_query_node(state: AdvancedRAGState) -> dict:
    question = state["question"]
    start = time.time()

    chain = ANALYZE_PROMPT | llm
    response = chain.invoke({"question": question})
    raw = response.content.strip()

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    try:
        parsed = json.loads(match.group()) if match else {}
    except json.JSONDecodeError:
        parsed = {}

    route = parsed.get("route", "rag")
    strategy = parsed.get("strategy", "basic")
    category = parsed.get("category", "general")
    elapsed = time.time() - start

    logs = state.get("logs", []) + [
        f"[analyze_query] route={route}, strategy={strategy}, category={category} ({elapsed:.2f}s)"
    ]
    metrics = dict(state.get("metrics", {}))
    metrics["analyze_time"] = elapsed

    print("\n=== [1] 질문 분석 (analyze_query) ===")
    print(f"원본 질문: {question}")
    print(f"분석 결과: route={route}, strategy={strategy}, category={category}")

    return {
        "route": route,
        "search_strategy": strategy,
        "category": category,
        "logs": logs,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# 노드 2. route_query : 조건부 엣지 함수 (analyze_query 이후 분기)
# ---------------------------------------------------------------------------
def route_after_analyze(state: AdvancedRAGState) -> str:
    if state.get("route") == "general":
        return "general_answer"

    strategy = state.get("search_strategy")
    if strategy in ("multi_query", "hybrid"):
        # hybrid도 먼저 multi_query_node를 거친 뒤 hyde_node로 이어집니다.
        return "multi_query"
    elif strategy == "hyde":
        return "hyde"
    else :
        return "basic_query"


def route_after_multi_query(state: AdvancedRAGState) -> str:
    if state.get("search_strategy") == "hybrid":
        return "hyde"
    return "retrieval"


# ---------------------------------------------------------------------------
# basic_query_node : 원본 질문을 그대로 검색 Query로 사용
# ---------------------------------------------------------------------------
def basic_query_node(state: AdvancedRAGState) -> dict:
    question = state["question"]
    logs = state.get("logs", []) + ["[basic_query] 원본 질문을 그대로 검색 Query로 사용"]

    print("\n=== [2] 검색 전략: Basic RAG (basic_query) ===")
    print(f"검색 Query: {question}")

    return {"search_queries": [question], "logs": logs}


# ---------------------------------------------------------------------------
# 노드 3. multi_query_node : 원본 질문을 3개의 다른 표현으로 확장
# ---------------------------------------------------------------------------
MULTI_QUERY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "사용자의 질문을 의미는 동일하지만 표현이 다른 3개의 검색 질문으로 변환하세요. "
            "각 질문은 한 줄에 하나씩만 작성하고, 번호나 부가 설명 없이 질문 문장만 출력하세요.",
        ),
        ("human", "질문: {question}"),
    ]
)


def multi_query_node(state: AdvancedRAGState) -> dict:
    question = state["question"]
    start = time.time()

    chain = MULTI_QUERY_PROMPT | llm
    response = chain.invoke({"question": question})
    lines = [
        line.strip("-•0123456789. ").strip()
        for line in response.content.strip().split("\n")
        if line.strip()
    ]
    queries = lines[:3]
    elapsed = time.time() - start

    existing = state.get("search_queries") or [question]
    search_queries = existing + queries

    logs = state.get("logs", []) + [
        f"[multi_query] {len(queries)}개 Query 생성 ({elapsed:.2f}s)"
    ]
    metrics = dict(state.get("metrics", {}))
    metrics["multi_query_time"] = elapsed

    print("\n=== [3] Multi-Query 생성 결과 (multi_query) ===")
    for i, q in enumerate(queries, 1):
        print(f"  {i}. {q}")

    return {
        "multi_queries": queries,
        "search_queries": search_queries,
        "logs": logs,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# 노드 4. hyde_node : 가상의 답변 문서를 생성해 검색 Query로 사용
# ---------------------------------------------------------------------------
HYDE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "다음 질문에 대한 답이 될 만한 가상의 정부 지원사업 상담 문서를 2~3문장으로 작성하세요. "
            "실제 안내문처럼 사실적인 어투로 작성하되, 이 내용은 검색 확장을 위한 것이며 "
            "최종 답변으로 사용하지 않습니다.",
        ),
        ("human", "질문: {question}"),
    ]
)


def hyde_node(state: AdvancedRAGState) -> dict:
    question = state["question"]
    start = time.time()

    chain = HYDE_PROMPT | llm
    response = chain.invoke({"question": question})
    hyde_doc = response.content.strip()
    elapsed = time.time() - start

    existing = state.get("search_queries") or [question]
    search_queries = existing + [hyde_doc]

    logs = state.get("logs", []) + [f"[hyde] 가상 문서 생성 ({elapsed:.2f}s)"]
    metrics = dict(state.get("metrics", {}))
    metrics["hyde_time"] = elapsed

    print("\n=== [4] HyDE 가상 문서 (hyde) ===")
    print(hyde_doc)

    return {
        "hypothetical_document": hyde_doc,
        "search_queries": search_queries,
        "logs": logs,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# general_answer_node : 문서와 무관한 질문에 대한 일반 LLM 답변 (검색 생략)
# ---------------------------------------------------------------------------
GENERAL_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "당신은 친절한 AI 어시스턴트입니다. 사용자의 일반 지식 질문에 답하세요."),
        ("human", "{question}"),
    ]
)


def general_answer_node(state: AdvancedRAGState) -> dict:
    question = state["question"]
    start = time.time()

    chain = GENERAL_PROMPT | llm
    response = chain.invoke({"question": question})
    answer = response.content.strip()
    elapsed = time.time() - start

    logs = state.get("logs", []) + ["[general_answer] 문서 검색 없이 일반 LLM 답변 생성"]
    metrics = dict(state.get("metrics", {}))
    metrics["generation_time"] = elapsed

    print("\n=== 일반 LLM 답변 (문서 검색 생략) ===")
    print(answer)

    return {
        "answer": answer,
        "logs": logs,
        "metrics": metrics,
        "search_queries": [],
        "raw_documents": [],
        "processed_documents": [],
        "compressed_context": "",
    }


# ---------------------------------------------------------------------------
# 노드 6. retrieval_node : search_queries 전체로 Vector DB 검색
# ---------------------------------------------------------------------------
def retrieval_node(state: AdvancedRAGState) -> dict:
    search_queries = state.get("search_queries") or [state["question"]]
    start = time.time()

    raw_documents = []
    for q in search_queries:
        results = vectorstore.similarity_search_with_relevance_scores(q, k=TOP_K_PER_QUERY)
        for doc, score in results:
            raw_documents.append(
                {
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "unknown"),
                    "category": doc.metadata.get("category", "general"),
                    "score": score,
                    "query": q,
                }
            )
    elapsed = time.time() - start

    sources = sorted(set(d["source"] for d in raw_documents))
    logs = state.get("logs", []) + [
        f"[retrieval] Query {len(search_queries)}개 검색, 총 {len(raw_documents)}개 문서 검색 ({elapsed:.2f}s)"
    ]
    metrics = dict(state.get("metrics", {}))
    metrics["retrieval_time"] = elapsed
    metrics["num_search_queries"] = len(search_queries)
    metrics["num_raw_documents"] = len(raw_documents)

    print("\n=== [5] Vector DB 검색 결과 (retrieval) ===")
    print(f"검색 Query 수: {len(search_queries)}")
    print(f"검색된 전체 문서 수: {len(raw_documents)}")
    print("검색 출처:")
    for s in sources:
        print(f"  - {s}")

    return {"raw_documents": raw_documents, "logs": logs, "metrics": metrics}


# ---------------------------------------------------------------------------
# 노드 7. post_processing_node : 중복/저품질/무관 문서 제거 + 개수 제한
# ---------------------------------------------------------------------------
def post_processing_node(state: AdvancedRAGState) -> dict:
    raw_documents = state.get("raw_documents", [])
    category = state.get("category", "general")
    start = time.time()

    # 1단계: 중복 제거 (page_content 기준)
    seen = set()
    deduped = []
    for d in raw_documents:
        if d["content"] not in seen:
            seen.add(d["content"])
            deduped.append(d)

    # 2단계: 유사도(relevance_score) 기준 제거
    filtered_by_score = [d for d in deduped if d["score"] >= SCORE_THRESHOLD]
    if not filtered_by_score:
        # 전부 걸러지면 점수 상위 문서라도 유지 (안전장치)
        filtered_by_score = sorted(deduped, key=lambda d: d["score"], reverse=True)[:POST_PROCESS_LIMIT]

    # 3단계: Metadata 카테고리 필터링
    if category and category not in ("general", "none", ""):
        category_filtered = [d for d in filtered_by_score if d["category"] == category]
        if not category_filtered:
            # 매칭되는 카테고리 문서가 없으면 필터를 완화합니다.
            category_filtered = filtered_by_score
    else:
        category_filtered = filtered_by_score

    # 4단계: 문서 개수 제한 (Reranking으로 넘길 후보를 점수 높은 순으로 상위 N개만 유지)
    category_filtered.sort(key=lambda d: d["score"], reverse=True)
    processed_documents = category_filtered[:POST_PROCESS_LIMIT]

    elapsed = time.time() - start
    logs = state.get("logs", []) + [
        f"[post_processing] {len(raw_documents)}개 -> 중복제거 {len(deduped)}개 -> "
        f"점수필터 {len(filtered_by_score)}개 -> 카테고리필터 {len(category_filtered)}개 -> "
        f"최종 {len(processed_documents)}개 ({elapsed:.2f}s)"
    ]
    metrics = dict(state.get("metrics", {}))
    metrics["post_processing_time"] = elapsed
    metrics["num_unique_documents"] = len(deduped)
    metrics["num_processed_documents"] = len(processed_documents)

    print("\n=== [6] Post-Processing 결과 (post_processing) ===")
    print(
        f"원본 {len(raw_documents)}개 -> 중복 제거 {len(deduped)}개 -> "
        f"점수 필터(>= {SCORE_THRESHOLD}) {len(filtered_by_score)}개 -> "
        f"카테고리('{category}') 필터 {len(category_filtered)}개 -> "
        f"최종 {len(processed_documents)}개"
    )
    for d in processed_documents:
        print(f"  - [{d['source']}] score={d['score']:.3f} category={d['category']}")

    return {"processed_documents": processed_documents, "logs": logs, "metrics": metrics}


# ---------------------------------------------------------------------------
# 노드 8. reranking_node : Cross-Encoder로 문서를 재채점하여 최종 순위 결정
# ---------------------------------------------------------------------------
def reranking_node(state: AdvancedRAGState) -> dict:
    question = state["question"]
    candidates = state.get("processed_documents", [])
    start = time.time()

    print("\n=== [7] Reranking 결과 (reranking) ===")

    if not candidates:
        elapsed = time.time() - start
        logs = state.get("logs", []) + ["[reranking] 후보 문서 없음, Reranking 생략"]
        print("후보 문서가 없어 Reranking을 생략합니다.")
        return {"reranked_documents": [], "logs": logs}

    if reranker_model is None:
        # Reranker 초기화에 실패한 경우, Post-Processing 점수 순서를 그대로 사용 (안전장치)
        reranked = sorted(candidates, key=lambda d: d["score"], reverse=True)[:RERANK_TOP_N]
        for d in reranked:
            d["rerank_score"] = None
        elapsed = time.time() - start

        print("⚠️ Reranker가 초기화되지 않아 Post-Processing 점수 순서를 그대로 사용합니다.")
        for d in reranked:
            print(f"  - [{d['source']}] score={d['score']:.3f}")

        logs = state.get("logs", []) + [
            f"[reranking] Reranker 미사용(초기화 실패), 기존 점수 순서 유지 ({elapsed:.2f}s)"
        ]
        metrics = dict(state.get("metrics", {}))
        metrics["reranking_time"] = elapsed
        metrics["num_reranked_documents"] = len(reranked)
        return {"reranked_documents": reranked, "logs": logs, "metrics": metrics}

    # 1. 채점을 위해 [질문, 문서내용] 형태의 짝(Pair)을 만듭니다.
    pairs = [[question, d["content"]] for d in candidates]
    # 2. Cross-Encoder 모델에게 적합도 점수 예측을 지시합니다.
    scores = reranker_model.predict(pairs)
    for d, s in zip(candidates, scores):
        d["rerank_score"] = float(s)

    # 3. 점수가 높은 순(내림차순)으로 정렬 후 상위 N개만 최종 통과시킵니다.
    reranked = sorted(candidates, key=lambda d: d["rerank_score"], reverse=True)[:RERANK_TOP_N]
    elapsed = time.time() - start

    print(f"Cross-Encoder 모델: {RERANK_MODEL_NAME}")
    print(f"후보 {len(candidates)}개 -> Reranking 후 상위 {len(reranked)}개")
    for d in reranked:
        print(f"  - [{d['source']}] rerank_score={d['rerank_score']:.4f} (기존 유사도 score={d['score']:.3f})")

    logs = state.get("logs", []) + [
        f"[reranking] Cross-Encoder로 {len(candidates)}개 문서 재채점, 상위 {len(reranked)}개 선택 ({elapsed:.2f}s)"
    ]
    metrics = dict(state.get("metrics", {}))
    metrics["reranking_time"] = elapsed
    metrics["num_reranked_documents"] = len(reranked)

    return {"reranked_documents": reranked, "logs": logs, "metrics": metrics}


# ---------------------------------------------------------------------------
# 노드 9. context_compression_node : 질문에 필요한 문장만 추출
# ---------------------------------------------------------------------------
COMPRESSION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "다음 문서에서 사용자의 질문에 답하는 데 필요한 문장만 추출하세요.\n"
            "규칙:\n"
            "1. 문서에 없는 내용을 추가하지 마세요.\n"
            "2. 사업명,지원내역, 지원대상, 예산, 사업 공고일을 포함하십시요.\n"
            "3. 각 문장(또는 문단) 앞에 [파일명]을 표시하세요.\n"
            "4. 질문과 관계없는 내용은 제거하세요.",
        ),
        ("human", "질문: {question}\n\n문서 목록:\n{documents}"),
    ]
)


def context_compression_node(state: AdvancedRAGState) -> dict:
    question = state["question"]
    # Reranking을 통과한 최종 상위 문서만 압축 대상으로 사용합니다.
    reranked_documents = state.get("reranked_documents", [])

    docs_text = "\n\n".join(f"[{d['source']}]\n{d['content']}" for d in reranked_documents)
    before_len = len(docs_text)

    start = time.time()
    chain = COMPRESSION_PROMPT | llm
    response = chain.invoke({"question": question, "documents": docs_text})
    compressed_context = response.content.strip()
    elapsed = time.time() - start
    after_len = len(compressed_context)

    logs = state.get("logs", []) + [
        f"[context_compression] {before_len}자 -> {after_len}자 ({elapsed:.2f}s)"
    ]
    metrics = dict(state.get("metrics", {}))
    metrics["compression_time"] = elapsed
    metrics["context_len_before"] = before_len
    metrics["context_len_after"] = after_len

    print("\n=== [8] Context Compression (context_compression) ===")
    print(f"압축 전 길이: {before_len}자")
    print(f"압축 후 길이: {after_len}자")
    print("압축된 Context:")
    print(compressed_context)

    return {"compressed_context": compressed_context, "logs": logs, "metrics": metrics}


# ---------------------------------------------------------------------------
# 노드 10. generate_answer_node : 압축된 Context로 최종 답변 생성
# ---------------------------------------------------------------------------
ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "제공된 참고 문서만 사용하여 답변하세요. " +
            "문서에서 확인할 수 없는 내용은 \"제공된 문서에서 확인할 수 없습니다.\"라고 답하세요." +
            "사업명,지원내역, 지원대상, 예산, 사업 공고일은 기본적으로 포함하세요. " +
            "답변 뒤에는 사용한 파일명을 '출처:' 목록으로 표시하세요.",
        ),
        ("human", "질문: {question}\n\n참고 문서:\n{context}"),
    ]
)


def generate_answer_node(state: AdvancedRAGState) -> dict:
    question = state["question"]
    context = state.get("compressed_context", "")

    start = time.time()
    chain = ANSWER_PROMPT | llm
    response = chain.invoke({"question": question, "context": context})
    answer = response.content.strip()
    elapsed = time.time() - start

    logs = state.get("logs", []) + [f"[generate_answer] 답변 생성 ({elapsed:.2f}s)"]
    metrics = dict(state.get("metrics", {}))
    metrics["generation_time"] = elapsed

    print("\n=== [9] 최종 답변 (generate_answer) ===")
    print(answer)

    return {"answer": answer, "logs": logs, "metrics": metrics}


# ---------------------------------------------------------------------------
# 노드 11. show_result_node : 전체 중간 결과 요약 출력 + total_time 계산
# ---------------------------------------------------------------------------
def show_result_node(state: AdvancedRAGState) -> dict:
    metrics = dict(state.get("metrics", {}))
    total_time = sum(v for k, v in metrics.items() if k.endswith("_time"))
    metrics["total_time"] = total_time

    print("\n" + "=" * 70)
    print("[show_result] 전체 실행 결과 요약")
    print("=" * 70)
    print(f" 1. 원본 질문          : {state.get('question')}")
    print(f" 2. 라우팅 결과        : {state.get('route')}")
    print(f" 3. 선택된 검색 전략   : {state.get('search_strategy')}")
    print(f" 4. Multi-Query 결과   : {state.get('multi_queries') or '(생성 안함)'}")
    print(f" 5. HyDE 가상 문서     : {state.get('hypothetical_document') or '(생성 안함)'}")
    print(f" 6. 검색된 전체 문서 수: {len(state.get('raw_documents') or [])}")
    print(f" 7. Post-Processing 후 문서 수: {len(state.get('processed_documents') or [])}")
    print(f" 8. Reranking 후 문서 수: {len(state.get('reranked_documents') or [])}")
    print(f" 9. 압축 전 Context 길이: {metrics.get('context_len_before', 0)}자")
    print(f"10. 압축 후 Context 길이: {metrics.get('context_len_after', 0)}자")
    print(f"11. 최종 답변:\n{state.get('answer')}")
    print(f"12. 전체 실행시간      : {total_time:.2f}초")
    print("=" * 70)

    return {"metrics": metrics}




# ---------------------------------------------------------------------------
# LangGraph 구성
# ---------------------------------------------------------------------------
def build_graph():
    graph = StateGraph(AdvancedRAGState)

    graph.add_node("analyze_query", analyze_query_node)
    graph.add_node("basic_query", basic_query_node)
    graph.add_node("multi_query", multi_query_node)
    graph.add_node("hyde", hyde_node)
    graph.add_node("general_answer", general_answer_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("post_processing", post_processing_node)
    graph.add_node("reranking", reranking_node)
    graph.add_node("context_compression", context_compression_node)
    graph.add_node("generate_answer", generate_answer_node)
    graph.add_node("show_result", show_result_node)

    graph.add_edge(START, "analyze_query")

    graph.add_conditional_edges(
        "analyze_query",
        route_after_analyze,
        {
            "basic_query": "basic_query",
            "multi_query": "multi_query",
            "hyde": "hyde",
            "general_answer": "general_answer",
        },
    )

    # hybrid 전략은 multi_query_node를 거친 뒤 hyde_node로 이어집니다.
    graph.add_conditional_edges(
        "multi_query",
        route_after_multi_query,
        {
            "hyde": "hyde",
            "retrieval": "retrieval",
        },
    )

    graph.add_edge("basic_query", "retrieval")
    graph.add_edge("hyde", "retrieval")
    graph.add_edge("general_answer", "show_result")

    graph.add_edge("retrieval", "post_processing")
    graph.add_edge("post_processing", "reranking")
    graph.add_edge("reranking", "context_compression")
    graph.add_edge("context_compression", "generate_answer")
    graph.add_edge("generate_answer", "show_result")
    graph.add_edge("show_result", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# 모듈 로딩 시 Graph를 한 번만 컴파일합니다.
# main.py(FastAPI)는 이 rag_graph 객체를 그대로 import 해서 사용합니다.
# (FastAPI의 `app = FastAPI()`와 이름 충돌을 피하기 위해 `app` 대신 `rag_graph` 사용)
# ---------------------------------------------------------------------------
rag_graph = build_graph()


# ---------------------------------------------------------------------------
# 단독 실행 시(python advanced_rag_agent.py) 기존과 동일하게 터미널 테스트를 수행합니다.
# FastAPI 등 다른 모듈에서 import 될 때는 이 블록이 실행되지 않습니다.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_questions = [
        "8월에 지원할 수 있는 사업을 알려주세요",  # 예상: basic
        "예비창업자나 스타트업이 할수 지원할 수 있는 사업을 알려주세요",  # 예상: multi_query
        "가장 예산이 많은 사업은?",  # 예상: hyde
        "녹색산업분야에서 고급인력이 성장 기반을 마련하기 위한 지원 사업을 추천해주세요.",  # 예상: hybrid
        "2027년 사업 시작은 언제부터 인가요?",  # 예상: general
    ]

    for i, q in enumerate(test_questions, 1):
        print("\n\n" + "#" * 70)
        print(f"# 테스트 질문 {i}: {q}")
        print("#" * 70)
        rag_graph.invoke(new_state(q))