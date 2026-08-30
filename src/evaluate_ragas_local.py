"""
evaluate_ragas_local.py
--------------------------------
Advanced RAG(advanced_rag_agent.py)의 답변 품질을 RAGAS로 평가합니다.
예전 evaluate_ragas_local.py(AdvancedRAG_All_Graph2.py 기반)를 새 웹 구조에
맞게 옮기면서, 두 가지 실행 방식을 함께 지원하도록 정리했습니다.

1) --mode agent (기본값, 예전 스크립트와 동일한 방식)
   - FastAPI 서버 없이 advanced_rag_agent.rag_graph를 파이썬 프로세스 안에서
     직접 invoke() 합니다.
   - reranked_documents의 원문(context)을 그대로 얻을 수 있어 별도 API 응답
     스키마 변경 없이도 context_precision / context_recall / faithfulness를
     정확히 계산할 수 있습니다. 서버를 안 띄워도 되어 가장 간단합니다.

2) --mode api
   - 실제 배포 구조(Streamlit -> FastAPI -> LangGraph)를 그대로 사용해,
     uvicorn으로 띄운 FastAPI의 POST /rag/query 를 requests로 호출합니다.
   - "API가 실제로 반환하는 값"을 평가하므로 배포 전 회귀 테스트에 적합합니다.
   - 이 방식을 쓰려면 main.py를 먼저 실행해 두어야 하고, RAGResponse에 추가된
     contexts 필드(검색된 문서 원문 목록)를 사용합니다. contexts가 없던 예전
     버전의 main.py로는 이 모드를 쓸 수 없으니 schemas.py / main.py가
     contexts 필드를 반환하는지 먼저 확인하세요.

두 방식 모두 question / answer / contexts / ground_truth 스키마는 동일하므로
RAGAS 평가 로직(dataset 생성 이후)은 공통으로 사용합니다.

실행 예:
    python evaluate_ragas_local.py                   # agent 모드 (기본, 서버 불필요)
    python evaluate_ragas_local.py --mode api         # api 모드 (main.py 먼저 실행 필요)
    python evaluate_ragas_local.py --mode api --api-url http://127.0.0.1:8000

평가(채점) LLM: qwen2.5:7b (Ollama). H200 한 대를 여러 명이 나눠 쓰는 환경을
고려해 max_workers=1로 순차 평가하고, 질문 사이에 짧은 대기시간을 둡니다.
"""

from __future__ import annotations

import argparse
import sys
import time
import types

# --- [패치] RAGAS 내부의 누락된 VertexAI 모듈 임포트 에러 우회 ---
try:
    import langchain_community.chat_models.vertexai
except ImportError:
    dummy_chat = types.ModuleType("langchain_community.chat_models.vertexai")
    dummy_chat.ChatVertexAI = type("ChatVertexAI", (object,), {})
    sys.modules["langchain_community.chat_models.vertexai"] = dummy_chat

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig  # 타임아웃 방지용 설정 모듈

from langchain_ollama import ChatOllama, OllamaEmbeddings

# ---------------------------------------------------------------------------
# 평가(채점) 모델 설정 (로컬 Ollama)
# ---------------------------------------------------------------------------
EVAL_LLM_MODEL = "qwen2.5:14b"   # 검증 전용 LLM. H200을 여러 명이 공유하는 환경을 고려해 9B로 고정.
EMBEDDING_MODEL = "bge-m3"      # advanced_rag_agent.py와 동일한 임베딩 모델을 재사용(추가 VRAM 점유 최소화)

# 질문 사이 대기 시간(초). 같은 GPU를 여러 명이 나눠 쓰는 상황에서 연속 요청으로
# 인한 부하/크래시(예: 이전에 겪은 Ollama CUDA 크래시)를 줄이기 위한 안전장치입니다.
# 필요 없으면 0으로 바꾸세요.
SLEEP_BETWEEN_QUESTIONS = 2.0

DEFAULT_API_URL = "http://127.0.0.1:8014"


# ---------------------------------------------------------------------------
# 평가 대상 질문 5개 + 정답(ground_truth)
# travel_data의 5개 카테고리(취소/여행상품/호텔/항공권/일정변경)를 1개씩 커버합니다.
# ground_truth는 각 문서 원문(및 [상담 요약])을 근거로 다시 작성한 값입니다.
# (예전 버전은 질문 2개뿐이었고, 취소 수수료 ground_truth가 "10%"로 잘못 적혀 있었습니다.
#  실제 취소특별약관.txt 기준 출발 1~7일 전 취소 수수료는 "30%"(성수기 40%)입니다.)
# ---------------------------------------------------------------------------
eval_questions = [
    "창업 3년 이하 기업이 지원 할 수 있는 사업을 알려주세요",                                  # 예상: basic
    "녹색산업분야에서 IT와 에너지 기술 기반을 마련하기 위한 지원 사업을 추천해주세요.",           # 예상: multi_query
    "가장 예산이 많은 사업은?",  # 예상: hyde

]

ground_truths = [
    """2 ㆍ초기창업패키지
유망 창업 아이템 및 기술을
보유한 초기 창업기업의 
사업화 지원을 통한 안정적인
시장 진입 및 성장 도모
①사업화 자금
②창업프로그램
업력 3년 이내 
초기창업기업 559 ’25.12월
중소벤처
기업부
(신산업
기술창업과)
창업진흥원
(초기도약실)""",
"""
11 ㆍ에코스타트업 지
원사업
녹색산업분야 유망 창업 
아이템이 있는 예비창업자
와 창업기업의 아이디
어, 기술의 사업화를 위한 
창업 성장 지원
①사업화 자금
②역량강화 프로
그램 지원
①예비창업자
②창업기업(업력 7년 이내)
③39세 이하의 예비
창업자
④39세 이하가 대표인
3년 이내의 창업기업
⑤그 외 특화창업기업
218 ’26.1월
기후에너지
환경부
(탈탄소녹색
산업혁신과)
한국환경산업
기술원
(창업사업화실) -
""",
    "'딥테크 밸류업 프로그램'이며, 이 사업의 예산은 84억원입니다.",

]

assert len(eval_questions) == len(ground_truths) == 5

# 결과 요약 표에 쓸 짧은 한글 라벨 (eval_questions와 순서가 같아야 합니다)
QUESTION_LABELS = [
    "취소 수수료",
    "오사카 추천",
    "호텔 추천",
    "항공권 수하물",
    "일정 변경",
]
assert len(QUESTION_LABELS) == len(eval_questions)


def build_summary_table(result) -> pd.DataFrame:
    """RAGAS 평가 결과를 'Question / Context Precision / Context Recall /
    Faithfulness / Answer Relevancy' 형태의 요약 표로 정리합니다.
    (질문 원문 대신 짧은 라벨을 쓰고, 점수는 소수점 둘째 자리까지 반올림)
    """
    df = result.to_pandas()

    metric_columns = {
        "context_precision": "Context Precision",
        "context_recall": "Context Recall",
        "faithfulness": "Faithfulness",
        "answer_relevancy": "Answer Relevancy",
    }
    missing = [c for c in metric_columns if c not in df.columns]
    if missing:
        print(f"⚠️ 결과에 다음 지표 컬럼이 없습니다(타임아웃/에러로 계산 실패 가능): {missing}")

    summary = pd.DataFrame({"Question": QUESTION_LABELS[: len(df)]})
    for raw_name, display_name in metric_columns.items():
        summary[display_name] = df[raw_name].round(2) if raw_name in df.columns else None

    return summary


# ---------------------------------------------------------------------------
# 모드 1) agent: FastAPI 없이 LangGraph를 직접 invoke (예전 스크립트와 동일한 방식)
# ---------------------------------------------------------------------------
def run_via_agent() -> tuple[list[str], list[list[str]]]:
    from advanced_rag_agent import new_state, rag_graph

    answers: list[str] = []
    contexts_list: list[list[str]] = []

    for i, q in enumerate(eval_questions, 1):
        print(f"\n[{i}/{len(eval_questions)}] [Agent 직접 실행]: {q}")
        state = new_state(q)
        result_state = rag_graph.invoke(state)

        answers.append(result_state.get("answer", ""))

        docs = result_state.get("reranked_documents", [])
        contexts_list.append([d["content"] for d in docs])

        if SLEEP_BETWEEN_QUESTIONS and i < len(eval_questions):
            time.sleep(SLEEP_BETWEEN_QUESTIONS)

    return answers, contexts_list


# ---------------------------------------------------------------------------
# 모드 2) api: 배포된 FastAPI(main.py)의 POST /rag/query 를 그대로 호출
# ---------------------------------------------------------------------------
def run_via_api(api_url: str) -> tuple[list[str], list[list[str]]]:
    import requests

    endpoint = f"{api_url.rstrip('/')}/rag/query"
    answers: list[str] = []
    contexts_list: list[list[str]] = []

    for i, q in enumerate(eval_questions, 1):
        print(f"\n[{i}/{len(eval_questions)}] [API 호출]: {q}  ({endpoint})")
        try:
            response = requests.post(endpoint, json={"question": q}, timeout=180)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(
                f"FastAPI 호출 실패: {e}\n"
                f"main.py가 실행 중인지 확인하세요 (uvicorn main:app --reload)."
            ) from e

        data = response.json()
        answers.append(data.get("answer", ""))
        # schemas.py / main.py에 추가된 contexts 필드(검색된 문서 원문 목록)를 사용합니다.
        contexts = data.get("contexts") or []
        if not contexts:
            print(
                "  ⚠️ 응답에 contexts가 비어 있습니다. main.py가 최신 버전인지 "
                "(RAGResponse.contexts 반환) 확인하세요."
            )
        contexts_list.append(contexts)

        if SLEEP_BETWEEN_QUESTIONS and i < len(eval_questions):
            time.sleep(SLEEP_BETWEEN_QUESTIONS)

    return answers, contexts_list


def main():
    parser = argparse.ArgumentParser(description="Advanced RAG를 RAGAS로 평가합니다.")
    parser.add_argument(
        "--mode",
        choices=["agent", "api"],
        default="agent",
        help="agent: LangGraph 직접 invoke (기본값, 서버 불필요) / api: FastAPI(main.py) 호출",
    )
    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help=f"--mode api일 때 사용할 FastAPI 서버 주소 (기본값: {DEFAULT_API_URL})",
    )
    args = parser.parse_args()

    print("=" * 60)
    print(f"RAGAS 평가 수집 시작 (실행 모드: {args.mode}, 평가 모델: {EVAL_LLM_MODEL})")
    print("=" * 60)

    if args.mode == "agent":
        answers, contexts_list = run_via_agent()
    else:
        answers, contexts_list = run_via_api(args.api_url)

    dataset = Dataset.from_dict(
        {
            "question": eval_questions,
            "answer": answers,
            "contexts": contexts_list,
            "ground_truth": ground_truths,
        }
    )

    print("\n" + "=" * 60)
    print("RAGAS 점수 계산 중 (로컬 LLM 채점)...")
    print("=" * 60)

    evaluator_llm = LangchainLLMWrapper(
        ChatOllama(model=EVAL_LLM_MODEL, temperature=0)
    )
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(model=EMBEDDING_MODEL)
    )

    # Ollama 부하를 줄이기 위한 실행 설정: 워커 1개로 순차 실행 + 공유 GPU를
    # 고려해 타임아웃을 넉넉히(600초) 잡습니다.
    ragas_config = RunConfig(timeout=600, max_workers=1)

    result = evaluate(
        dataset=dataset,
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=ragas_config,
        raise_exceptions=False,  # 타임아웃 발생 시에도 프로그램 종료 방지
    )

    print("\n[평가 결과 요약]")
    print(result)

    print("\n[세부 항목별 결과]")
    df = result.to_pandas()
    print(df.to_string())

    print("\n[요약 표]")
    summary_df = build_summary_table(result)
    print(summary_df.to_string(index=False))

    # 표를 다시 쓸 수 있도록 CSV로도 저장합니다.
    summary_path = "ragas_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\n요약 표를 '{summary_path}'로 저장했습니다.")


if __name__ == "__main__":
    main()
