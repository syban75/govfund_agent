"""정부 지원사업 PDF를 Chroma 벡터 DB로 구축한다.

원본 ``govfund_BuildDB.py``의 실행 위치 의존 경로, PDF 텍스트 열기,
재실행 시 중복 저장 문제를 보완한 버전이다.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_DIR / "rag_data"
DEFAULT_DB_DIR = PROJECT_DIR / "chroma_govfund_db"

COLLECTION_NAME = "govfund_guide"
EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "bge-m3")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://10.8.0.1:11434")
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
BATCH_SIZE = 100

CATEGORY_MAP = {
    "(공고문)_2026년도_중앙부처_및_지자체_창업지원사업_통합공고문(제2025-648호,_2025.12.19.).pdf": "notice",
}


def load_documents(data_dir: Path) -> list[Document]:
    """data_dir의 모든 PDF를 페이지 단위 Document로 읽는다."""
    pdf_files = sorted(data_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {data_dir}")

    documents: list[Document] = []
    for path in pdf_files:
        pdf_docs = PyPDFLoader(str(path)).load()
        category = CATEGORY_MAP.get(path.name, "general")
        valid_docs = []
        for doc in pdf_docs:
            if not doc.page_content.strip():
                continue
            doc.metadata.update(
                {
                    "source": path.name,
                    "source_path": str(path.resolve()),
                    "category": category,
                }
            )
            valid_docs.append(doc)

        documents.extend(valid_docs)
        total_chars = sum(len(doc.page_content) for doc in valid_docs)
        print(
            f"- 로드 완료: {path.name} "
            f"({len(valid_docs)}페이지, {total_chars:,}자, category={category})"
        )

    if not documents:
        raise ValueError("PDF에서 추출된 텍스트가 없습니다. 스캔 PDF라면 OCR이 필요합니다.")
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """문서를 검색용 청크로 분할하고 출처별 청크 번호를 기록한다."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = splitter.split_documents(documents)

    source_counts: dict[str, int] = {}
    for chunk in chunks:
        source = str(chunk.metadata.get("source", "unknown"))
        chunk_index = source_counts.get(source, 0)
        chunk.metadata["chunk_index"] = chunk_index
        source_counts[source] = chunk_index + 1

    print(f"- 분할 완료: {len(documents)}페이지 -> {len(chunks)}청크")
    return chunks


def make_chunk_id(chunk: Document) -> str:
    """동일 문서 재실행 시 덮어쓸 수 있는 결정적 ID를 만든다."""
    identity = "\n".join(
        [
            str(chunk.metadata.get("source", "")),
            str(chunk.metadata.get("page", "")),
            str(chunk.metadata.get("chunk_index", "")),
            chunk.page_content,
        ]
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def build_vector_db(chunks: list[Document], db_dir: Path, reset: bool = False) -> Chroma:
    """청크를 Chroma에 배치 단위로 upsert한다."""
    db_dir.mkdir(parents=True, exist_ok=True)
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(db_dir),
    )

    if reset:
        try:
            vectorstore.delete_collection()
        except ValueError:
            pass
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(db_dir),
        )

    ids = [make_chunk_id(chunk) for chunk in chunks]
    for start in range(0, len(chunks), BATCH_SIZE):
        end = start + BATCH_SIZE
        vectorstore.add_documents(chunks[start:end], ids=ids[start:end])
        print(f"- 저장 진행: {min(end, len(chunks))}/{len(chunks)}")
    return vectorstore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="정부 지원사업 PDF 벡터 DB 구축")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
    parser.add_argument(
        "--reset",
        action="store_true",
        help="기존 컬렉션을 삭제한 후 새로 구축합니다.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    db_dir = args.db_dir.resolve()

    if CHUNK_OVERLAP >= CHUNK_SIZE:
        raise ValueError("CHUNK_OVERLAP은 CHUNK_SIZE보다 작아야 합니다.")

    print(f"데이터 경로: {data_dir}")
    print(f"DB 경로: {db_dir}")
    print(f"임베딩: {EMBEDDING_MODEL} ({OLLAMA_BASE_URL})")
    documents = load_documents(data_dir)
    chunks = split_documents(documents)
    build_vector_db(chunks, db_dir, reset=args.reset)
    print(f"저장 완료: collection={COLLECTION_NAME}, chunks={len(chunks)}")


if __name__ == "__main__":
    main()
