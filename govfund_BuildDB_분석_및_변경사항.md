# govfund_BuildDB.py 분석 및 변경사항

## 분석 결과

1. `PyPDFLoader`를 호출하기 전에 PDF를 `open(path, "r", encoding="utf-8")`로 엽니다. PDF는 바이너리 파일이므로 `UnicodeDecodeError`가 발생할 수 있으며, 이 파일 객체는 실제 로딩에도 사용되지 않습니다.
2. `../rag_data`, `../chroma_govfund_db`가 현재 작업 디렉터리를 기준으로 해석됩니다. 프로젝트 루트에서 실행하면 프로젝트 바깥 경로를 참조하므로 실행 위치에 따라 결과가 달라집니다.
3. 기존 컬렉션에 `Chroma.from_documents()`를 반복 호출하므로 재실행할 때 같은 문서가 다른 자동 ID로 중복 저장될 수 있습니다.
4. 빈 페이지도 청크 처리 대상에 들어가며, 스캔 PDF처럼 텍스트가 전혀 추출되지 않는 경우 원인을 명확히 안내하지 않습니다.
5. 모든 청크를 한 번에 임베딩 요청하므로 문서량이 커질 때 요청 크기와 장애 복구 측면에서 불리합니다.
6. Ollama 주소가 코드에 고정되어 환경별 변경이 번거롭습니다.
7. 전역 `chunk_index`만 기록해 출처별 청크 순서를 파악하기 어렵습니다.

## 새 파일의 변경 내용

- 원본 파일은 변경하지 않고 `src/govfund_BuildDB_new.py`를 추가했습니다.
- `__file__` 기준으로 프로젝트 경로를 계산해 어느 위치에서 실행해도 기본 경로가 같습니다.
- 불필요하고 잘못된 PDF 텍스트 모드 열기를 제거했습니다.
- 빈 페이지를 제외하고, 추출 텍스트가 전혀 없으면 OCR 필요 가능성을 안내합니다.
- 파일명·페이지·청크 번호·본문으로 SHA-256 ID를 생성하여 같은 입력의 재실행이 중복되지 않게 했습니다.
- 임베딩 및 저장을 100개 단위로 처리합니다.
- `OLLAMA_BASE_URL`, `OLLAMA_EMBEDDING_MODEL` 환경변수와 `--data-dir`, `--db-dir` 옵션을 지원합니다.
- 기존 컬렉션을 명시적으로 새로 만들 때만 `--reset`을 사용하도록 했습니다.

## 실행 방법

```powershell
python src/govfund_BuildDB_new.py
```

기존 컬렉션을 삭제하고 완전히 다시 구축하려면 다음과 같이 실행합니다.

```powershell
python src/govfund_BuildDB_new.py --reset
```

`--reset`은 해당 DB의 `govfund_guide` 컬렉션을 삭제하므로 필요한 경우에만 사용해야 합니다.

## 호환성 참고

현재 `src/govfund_agent.py`는 `chroma_travel_db`의 `travel_guide` 컬렉션을 참조합니다. 빌드 스크립트의 기본값인 `chroma_govfund_db`와 `govfund_guide`를 사용하려면 에이전트 측 설정도 별도로 일치시켜야 합니다. 이번 작업에서는 요청 범위와 원본 비변경 원칙에 따라 에이전트 소스를 수정하지 않았습니다.
