"""
govfund_BuildDB.py
--------------------------------
정부지원사업문서 (../rag_data)의 pdf를 로드하여
text으로 전환하고
Chunk 분할 -> Embedding -> Chroma Vector DB 저장까지 수행하는 스크립트입니다.

담당 기능:
1. pdf 문서 로딩 및 txt 전환
2. Document 생성 (파일명/카테고리 metadata 부여)
3. Chunk 분할
4. Embedding 생성
5. Chroma Vector DB 저장

실행:
    python govfund_BuildDB.py

사전 준비:
- Ollama 로컬 서버 실행 (ollama serve) 및 임베딩 모델 다운로드: ollama pull bge-m3
- pip install -r requirements.txt
- rag_data에 pdf

이 스크립트를 먼저 실행해 Vector DB(chroma_govfund_db 폴더)를 만들어 두면,
govfund_agent.py 는 DB를 다시 만들지 않고 검색만 수행합니다.
"""

from __future__ import annotations

import os
import glob

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


# ---------------------------------------------------------------------------
# 설정값
# ---------------------------------------------------------------------------
DATA_DIR = "../rag_data"
DB_DIR = "../chroma_govfund_db"
COLLECTION_NAME = "govfund_guide"
EMBEDDING_MODEL = "bge-m3"  # 사용자 환경에 맞게 변경 (ollama pull bge-m3 필요)
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50

# 파일명 -> 카테고리 매핑 (post_processing / routing 에서 함께 사용)
CATEGORY_MAP = {
    "(공고문)_2026년도_중앙부처_및_지자체_창업지원사업_통합공고문(제2025-648호,_2025.12.19.).pdf": "notice",
}


# ---------------------------------------------------------------------------
# 1. pdf 문서 로딩 -> Document 생성
# ---------------------------------------------------------------------------
def load_documents() -> list[Document]:
    documents: list[Document] = []
    pdf_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.pdf")))

    if not pdf_files:
        raise FileNotFoundError(
            f"'{DATA_DIR}' 폴더에서 pdf 파일을 찾을 수 없습니다. "
            "rag_data 폴더 위치를 확인하세요."
        )

    for path in pdf_files:
        filename = os.path.basename(path)
        with open(path, "r", encoding="utf-8") as f:
            loader_notice = PyPDFLoader(path)

            pdf_docs = loader_notice.load()

            category = CATEGORY_MAP.get(filename, "general")

            for doc in pdf_docs:
                doc.metadata["category"] = category

            documents.extend(pdf_docs)

            total_chars = sum(len(doc.page_content) for doc in pdf_docs)

            print(f"- 로드 완료: {filename} ({total_chars:,}자, category={category})")

    return documents


# ---------------------------------------------------------------------------
# 2. Chunk 분할
# ---------------------------------------------------------------------------
def split_documents(documents: list[Document]) -> list[Document]:

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    chunks = splitter.split_documents(documents)

    # chunk 순번 metadata 부여 (디버깅/추적용)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    print(f"-splitting : docs: {len(documents)} => chunks : {len(chunks)}")

    return chunks


# ---------------------------------------------------------------------------
# 3~4. Embedding 생성 + Chroma Vector DB 저장
# ---------------------------------------------------------------------------
def build_vector_db(chunks: list[Document]):
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url="http://10.8.0.1:11434")

    if os.path.exists(DB_DIR):
        print(f"\n기존 '{DB_DIR}' 폴더가 존재합니다. 동일 컬렉션을 다시 생성/덮어씁니다.")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=DB_DIR,
    )

    return vectorstore


def main():
    print("=" * 60)
    print("[1] PDF 문서 로딩")
    print("=" * 60)
    documents = load_documents()
    print(f"총 {len(documents)}개 문서 로드 완료\n")

    print("=" * 60)
    print("[2] Chunk 분할")
    print("=" * 60)
    chunks = split_documents(documents)
    print(f"총 {len(chunks)}개 Chunk 생성")
    for c in chunks[:3]:
        preview = c.page_content[:60].replace("\n", " ")
        print(f"  예시 - [{c.metadata['source']}] {preview}...")

    print("\n" + "=" * 60)
    print("[3] Embedding 생성 및 Chroma Vector DB 저장")
    print("=" * 60)
    build_vector_db(chunks)
    print(f"'{DB_DIR}' 폴더에 '{COLLECTION_NAME}' 컬렉션으로 저장 완료")
    print("\n이제 AdvancedRAG_All_Graph.py 를 실행해 RAG 파이프라인을 테스트할 수 있습니다.")


if __name__ == "__main__":
    main()
