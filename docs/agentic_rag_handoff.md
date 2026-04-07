# Agentic RAG Handoff

이 문서는 `SK On 분석 에이전트`, `CATL 분석 에이전트`, 이후 `SWOT 비교 에이전트`, `보고서 생성 에이전트`가 공통으로 사용하는 RAG 레이어를 어떻게 쓰는지 설명합니다.

## 1. 이 RAG가 하는 일

- PDF 4종을 청크로 나눠 벡터 인덱스로 저장합니다.
- 질문이 들어오면 컬렉션에 맞는 검색 전략을 자동으로 세웁니다.
- 1차 검색 후 필요한 출처가 비어 있으면 질의를 바꿔 다시 검색합니다.
- 최종적으로 `근거 청크 + 출처 + 페이지`를 반환합니다.

현재 구현 기준으로 agentic 요소는 아래 두 가지입니다.

- `Planning`: 질문을 보고 질의를 여러 개로 확장
- `Reflection`: 검색 결과에 필요한 출처가 없으면 재검색

## 2. 컬렉션 선택 규칙

- `market_agent`
  - 시장 자료만 필요할 때 사용
  - 사용 문서: `market_report.pdf`, `analyst_report.pdf`
- `skon_agent`
  - SK온 기업 분석이 목적일 때 사용
  - 사용 문서: `skon.pdf` + 시장 자료 2종
- `catl_agent`
  - CATL 기업 분석이 목적일 때 사용
  - 사용 문서: `catl.pdf` + 시장 자료 2종
- `swot_agent`
  - SK온 vs CATL 비교 분석일 때 사용
  - 사용 문서: 전체 4종
- `report_agent`
  - 최종 보고서 생성용 근거 수집 시 사용
  - 사용 문서: 전체 4종

팀원용 핵심 규칙:

- SK온 에이전트는 무조건 `skon_agent`
- CATL 에이전트는 무조건 `catl_agent`
- 비교 분석은 `swot_agent`
- 최종 본문 생성은 `report_agent`

## 3. 먼저 해야 할 준비

프로젝트 루트에서:

```bash
uv sync
python app.py build-indices --table-backend auto
```

설명:

- `uv sync`: 패키지 설치
- `build-indices`: PDF를 읽고 FAISS 인덱스를 생성
- 표 추출은 현재 `CATL -> Camelot`, `SK온 -> pdfplumber`를 우선 사용

## 4. CLI로 직접 확인하는 방법

컬렉션 확인:

```bash
python app.py list-collections
```

표 추출 상태 확인:

```bash
python app.py inspect-tables --source catl.pdf --limit 3
python app.py inspect-tables --source skon.pdf --limit 3
```

기본 검색:

```bash
python app.py query --collection skon_agent --question "SK온의 생산능력과 해외 생산법인 현황은?"
```

Agentic RAG 검색:

```bash
python app.py agentic-query --collection skon_agent --question "SK온의 생산능력과 해외 생산법인 현황을 알려줘"
python app.py agentic-query --collection catl_agent --question "CATL의 2025년 재무 성과와 분기별 매출 흐름을 알려줘"
```

파이프라인용 state 생성:

```bash
python app.py build-state --collection skon_agent --question "SK온의 생산능력과 해외 생산법인 현황을 알려줘"
python app.py build-state --collection catl_agent --question "CATL의 2025년 재무 성과와 분기별 매출 흐름을 알려줘" --save-json outputs/catl_state.json --save-context outputs/catl_context.txt
```

## 5. 팀원이 코드에서 가져다 쓰는 방법

예시:

```python
from pathlib import Path

from rag import RAGConfig, agentic_similarity_search


PROJECT_ROOT = Path(__file__).resolve().parent

config = RAGConfig()

result = agentic_similarity_search(
    collection_name="skon_agent",
    question="SK온의 생산능력과 해외 생산법인 현황을 알려줘",
    config=config,
    project_root=PROJECT_ROOT,
    final_k=6,
    per_query_k=4,
    max_rounds=2,
)
```

반환값 구조:

- `result.final_hits`
  - 최종 근거 청크 목록
- `result.rounds`
  - 검색이 몇 번 돌았는지
- `result.sufficient`
  - 근거가 충분하다고 판단했는지
- `result.missing_sources`
  - 아직 비어 있는 출처가 있는지

실전에서는 아래 함수를 더 추천:

```python
from rag import RAGConfig, build_agent_retrieval_state

state = build_agent_retrieval_state(
    collection_name="skon_agent",
    question="SK온의 생산능력과 해외 생산법인 현황을 알려줘",
    config=RAGConfig(),
    project_root=PROJECT_ROOT,
)
```

이 state에는 아래가 포함됩니다.

- `state.coverage`
- `state.evidence_packets`
- `state.evidence_by_topic`
- `state.prompt_context`

## 6. 팀원 프롬프트에 넣는 방식

권장 방식:

1. 팀원 에이전트가 질문 생성
2. 해당 컬렉션으로 `build_agent_retrieval_state(...)` 호출
3. `state.prompt_context` 또는 `state.evidence_packets`를 프롬프트에 주입
4. LLM이 답변할 때 반드시 `source`, `page`를 인용

예시 포맷:

```text
[Evidence 1]
source: skon.pdf
page: 2
chunk_type: table
content:
...청크 내용...
```

LLM 프롬프트 규칙 권장:

- 주어진 근거 안에서만 답변할 것
- 수치가 있으면 source와 page를 반드시 표시할 것
- 근거가 부족하면 추측하지 말고 부족하다고 말할 것

## 7. SK온/CATL 팀원에게 이렇게 전달하면 됨

짧은 전달 문구 예시:

```text
SK온 분석은 `skon_agent`, CATL 분석은 `catl_agent` 컬렉션을 사용하면 됩니다.
질문을 넣으면 RAG가 먼저 질의를 확장해서 검색하고, 회사 자료/시장 자료가 부족하면 한 번 더 검색합니다.
반환되는 `final_hits`를 그대로 프롬프트의 evidence로 넣고, 답변에는 반드시 source/page를 인용해주세요.
인덱스가 없으면 먼저 `python app.py build-indices --table-backend auto`를 한 번 실행하면 됩니다.
```

## 8. 지금 구조의 한계

- 현재는 retrieval 단계가 agentic합니다.
- 아직 LLM 기반 `질문 재작성`, `근거 요약`, `응답 자기검토`는 붙어 있지 않습니다.
- 교수님이 이미지 수준의 full agentic RAG를 원하면 다음 단계로 아래가 추가되면 좋습니다.

추가 권장:

- 검색 planner LLM
- 답변 초안 생성기
- answer critic 또는 reflection node
- HITL 검토 노드

## 9. 추천 워크플로우

너의 담당 범위 기준:

1. 문서 추출 안정화
2. 컬렉션별 인덱스 생성
3. agentic retrieval 제공
4. 팀원은 retrieval 결과를 받아 기업 분석 작성
5. 이후 SWOT/보고서 에이전트가 같은 근거층을 재사용
