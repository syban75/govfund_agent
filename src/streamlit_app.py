"""
streamlit_app.py
--------------------------------
Streamlit Frontend - 정부 지원 사업 챗봇  Agent (SSE 스트리밍 버전)

Streamlit은 LangGraph를 직접 호출하지 않고, requests.post(..., stream=True)로
FastAPI(main.py) 의 POST /rag/query/stream 엔드포인트를 SSE로 소비합니다.

    Streamlit --requests.post(stream=True)--> FastAPI SSE
        analyze_query 완료 -> node_update 이벤트
        retrieval 완료     -> node_update 이벤트
        ...
        generate_answer 완료 -> node_update 이벤트
        show_result 완료     -> done 이벤트 (최종 답변 + metrics)

노드가 끝날 때마다 화면에 진행 상황이 즉시 표시되고, 마지막 done 이벤트가
오면 최종 답변과 RAG 처리 결과(route / strategy / category / source / 실행시간)를
렌더링합니다.

실행 (FastAPI 서버가 먼저 떠 있어야 합니다):
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
from typing import Any, Iterator

import requests
import streamlit as st

API_BASE = "http://127.0.0.1:8014"
STREAM_URL = f"{API_BASE}/rag/query/stream"


def iter_sse_events(response: requests.Response) -> Iterator[tuple[str, dict[str, Any]]]:
    """requests 스트리밍 응답에서 SSE "event / data" 쌍을 순서대로 뽑아냅니다.

    SSE 포맷:
        event: node_update
        data: {...json...}
        (빈 줄로 이벤트 구분)
    """
    event_name = "message"
    data_lines: list[str] = []

    for raw_line in response.iter_lines(decode_unicode=True):
        if raw_line is None:
            continue
        line = raw_line.rstrip("\r")

        if line == "":
            if data_lines:
                yield event_name, _parse_data("\n".join(data_lines))
            event_name = "message"
            data_lines = []
            continue

        if line.startswith("event:"):
            event_name = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:"):].strip())

    if data_lines:
        yield event_name, _parse_data("\n".join(data_lines))


def _parse_data(raw: str) -> dict[str, Any]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw}


st.set_page_config(page_title="정부 지원 사업 상담 Agent", page_icon="✈️", layout="centered")

st.title("✈️ 정부 지원 사업 상담 Agent")
st.caption("Streamlit → FastAPI(SSE) → LangGraph Advanced RAG → Chroma Vector DB")

st.divider()

# ---------------------------------------------------------------------------
# 질문 입력
# ---------------------------------------------------------------------------
st.subheader("질문")
question = st.text_area(
    "질문 입력",
    value="예비창업자나 스타트업이 할수 지원할 수 있는 사업을 알려주세요",
    height=80,
    label_visibility="collapsed",
)

ask_clicked = st.button("질문하기", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# SSE 스트림 소비 및 결과 표시
# ---------------------------------------------------------------------------
if ask_clicked:
    if not question.strip():
        st.warning("질문을 입력해주세요.")
    else:
        st.divider()
        st.subheader("실행 진행 상황")
        progress_placeholder = st.empty()
        steps: list[str] = []

        final_result: dict[str, Any] | None = None
        error_message: str | None = None

        try:
            with requests.post(
                STREAM_URL, json={"question": question}, stream=True, timeout=300
            ) as response:
                response.raise_for_status()
                for event_name, data in iter_sse_events(response):
                    if event_name == "node_update":
                        label = data.get("label", data.get("node", ""))
                        message = data.get("message")
                        line = f"- ✅ **{label}**"
                        if message:
                            line += f"  \n  &nbsp;&nbsp;&nbsp;&nbsp;{message}"
                        steps.append(line)
                        progress_placeholder.markdown("\n".join(steps))
                    elif event_name == "done":
                        final_result = data
                    elif event_name == "error":
                        error_message = data.get("message", "알 수 없는 오류가 발생했습니다.")
        except requests.exceptions.RequestException as e:
            error_message = f"API 호출 실패: {e}\n\nFastAPI 서버(uvicorn main:app --reload)가 실행 중인지 확인하세요."

        if error_message:
            st.error(f"RAG 파이프라인 실행 중 오류가 발생했습니다.\n\n{error_message}")

        elif final_result:
            st.divider()
            st.subheader("AI 답변")
            st.write(final_result.get("answer", "(답변 없음)"))

            st.divider()
            st.subheader("RAG 처리 결과")

            metrics = final_result.get("metrics", {})
            sources = final_result.get("sources", [])
            total_time = metrics.get("total_time", 0)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Route** : {final_result.get('route', '-')}")
                st.markdown(f"**Strategy** : {final_result.get('search_strategy', '-')}")
                st.markdown(f"**Category** : {final_result.get('category', '-')}")
            with col2:
                st.markdown(f"**Source** : {', '.join(sources) if sources else '-'}")
                st.markdown(f"**실행시간** : {total_time:.2f} sec")

            with st.expander("세부 Metrics 보기"):
                st.json(metrics)
