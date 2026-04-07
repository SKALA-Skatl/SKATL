# SKATL - 배터리 시장 전략 분석 보고서

전기차 chasm 여파 속 SK On과 CATL의 포트폴리오 다각화 전략을 Multi-Agent 기반으로 비교 분석하고, 의사결정에 활용 가능한 전략 분석 보고서를 자동 생성하는 시스템

## Overview

- **Objective** : 전기차 캐즘 장기화 환경에서 SK On(한국)과 CATL(중국)의 포트폴리오 다각화 전략을 객관적인 데이터 기반으로 비교 분석하여, 경영진·투자 의사결정자가 바로 활용 가능한 Comparative SWOT 전략 분석 보고서 생성
- **Method** : Distributed + Central Orchestrator 패턴 기반 Multi-Agent 시스템. Market Agent가 시장 환경을 분석하고, Central Orchestrator가 SKON/CATL Strategy Agent를 병렬(Fan-out) 실행한 뒤 결과를 병합(Fan-in). 각 핵심 판단 구간에 Human Review를 삽입해 품질을 제어하며, Comparative SWOT Agent와 Report Agent가 최종 보고서를 생성
- **Tools** : LangGraph, LangChain, OpenAI GPT-4o, FAISS, BAAI/bge-m3, Tavily Search, LangSmith

## Features

- **Agentic RAG** : Market · SK On · CATL Agent 내부에 검색 → 관련성 평가 → 쿼리 재작성 루프 적용 (max_rewrites: 3). FAISS IndexFlatIP + bge-m3 임베딩 기반으로 공식 IR 자료·산업 리포트에서 원문 컨텍스트를 그대로 수집
- **병렬 전략 분석** : Central Orchestrator가 SKON/CATL Strategy Agent를 Fan-out으로 병렬 실행하고, Fan-in으로 결과 병합. 분석 품질 부족 시 해당 Agent만 선택적 재실행 (max_retry: 2)
- **3단계 Human-in-the-Loop** : Review #1(시장 분석 후) · Review #2(전략 분석 후) · Review #3(보고서 생성 후). 각 단계에서 승인 또는 재조사 지시가 가능하며 LangGraph `interrupt()`로 구현
- **근거 기반 분석** : 모든 분석 축(캐즘 대응, 시장 포지션, 기술 포트폴리오, ESS 전략, 규제 리스크, 원가 구조)에 출처 ID·URL·신뢰도 스코어를 함께 기록. 요약 없이 원문 컨텍스트를 그대로 보존
- **확증 편향 방지** : 기업 홍보성 자료 편향을 차단하기 위해 외부 기사·산업 리포트·공식 IR 자료를 복수 출처로 교차 검증(cross_verified). 출처 신뢰도를 recency · source_tier · rag_relevance · cross_verified 4가지 기준으로 바이너리 평가하며, Comparative SWOT · Report Agent는 수집 결과만 사용(RAG 미적용)하여 과도한 해석을 방지
- **최신성 보장** : Tavily 기반 웹 검색으로 2025년 이후 최신 트렌드를 RAG 결과에 보완

## Tech Stack

| Category      | Details                                                              |
|---------------|----------------------------------------------------------------------|
| Framework     | LangGraph 1.1+, LangChain 1.2+, Python 3.11+                        |
| LLM           | GPT-4o via OpenAI API (전략 분석), GPT-4o-mini (RAG 쿼리 재작성)    |
| Retrieval     | FAISS (IndexFlatIP, cosine similarity ≥ 0.75)                        |
| Embedding     | BAAI/bge-m3 via langchain-huggingface (한·영 혼재 문서 대응)          |
| Web Search    | Tavily Search API                                                    |
| Observability | LangSmith (트레이싱, 노드별 실행 로그)                               |

## Agents

- **Market Agent** : 배터리 시장 환경 조사. EV 캐즘 현황·시장 점유율·LFP vs NCM 트렌드·ESS/HEV 성장성·IRA 규제·원가 경쟁력을 Agentic RAG + 웹 검색으로 수집하여 `MarketContext` 구조체로 출력
- **Central Orchestrator** : Market Agent 결과를 받아 SKON/CATL Strategy Agent를 병렬 실행하고 Fan-in으로 결과를 병합. 분석 충분성 판단 및 선택적 재실행 제어, Human Review #2 HITL 관리
- **SKON Strategy Agent** : 시장 맥락 기반 SK On 전략 조사. 공식 IR 자료(RAG)와 웹 검색을 결합해 6개 분석 축 원문 수집
- **CATL Strategy Agent** : SKON Agent와 동일 구조·동일 분석 축으로 CATL 전략을 독립 분석. 단일 템플릿(`run_strategy_agent`)으로 구현하여 동일 기준 비교를 보장
- **Comparative SWOT Agent** : SKON/CATL 분석 결과를 동일 프레임으로 비교. 원가 경쟁력·기술 경쟁력·시장 선점 가능성·지역 리스크·수익성 방어력 5개 항목 기반 Comparative SWOT 생성
- **Report Agent** : Summary · 시장 배경 · 기업별 전략 · Comparative SWOT · 종합 시사점 · Reference 포함 최종 보고서 생성. Human Review #3 이후 수정 재실행 가능

## Architecture

```
사용자 요청
    │
    ▼
┌─────────────────┐
│   Market Agent  │  Agentic RAG (배터리 시장 리포트, 산업 분석 보고서)
│  + web_search   │  → MarketContext 출력
└────────┬────────┘
         │
    Human Review #1 (충분성 · 조사 범위 조정)
         │ 승인
         ▼
┌──────────────────────┐
│ Central Orchestrator │  Fan-out (병렬, 최대 2회 재실행)
└──────┬───────────────┘
       │                    │
       ▼                    ▼
┌────────────────┐  ┌────────────────┐
│ SKON Strategy  │  │ CATL Strategy  │  각각 Agentic RAG + web_search
│     Agent      │  │     Agent      │  (SK On IR 자료 / CATL Annual Report)
└───────┬────────┘  └────────┬───────┘
        │                    │
        └──────────┬─────────┘
                   │ Fan-in
         Human Review #2 (충분성 · 비교 축 확인)
                   │ 승인
                   ▼
     ┌──────────────────────────┐
     │ Comparative SWOT Agent   │  RAG 미적용, 수집 결과만 사용
     └────────────┬─────────────┘
                  ▼
     ┌──────────────────────────┐
     │      Report Agent        │  최종 보고서 생성
     └────────────┬─────────────┘
                  │
         Human Review #3 (객관성 · 보고서 내용 확인)
                  │ 승인
                  ▼
       배터리 시장 전략 분석 보고서
```

## Report Structure

생성되는 보고서의 목차 구성:

1. **Summary** — 보고서 핵심 내용 요약
2. **시장 배경** — 배터리 시장 환경 변화 (캐즘 현황, 기술 트렌드, 규제 동향)
3. **기업별 포트폴리오 다각화 및 핵심 경쟁력** — SK On / CATL 각각
4. **Comparative SWOT** — 동일 기준 5개 항목 비교 평가
5. **종합 시사점** — 의사결정자 활용 가능한 전략적 시사점
6. **Reference** — 출처 목록 (source_id · URL · 신뢰도 스코어)

## Directory Structure

```
Skatl/
├── src/
│   ├── agents/
│   │   └── strategy_agent.py       # SKON/CATL 공통 실행 템플릿
│   ├── orchestrator/
│   │   └── orchestrator.py         # Central Orchestrator 그래프
│   ├── prompts/
│   │   └── strategy_prompt.py      # 회사별 분석 과제 프롬프트
│   ├── schemas/
│   │   ├── agent_io.py             # Agent 입출력 타입 정의
│   │   ├── confidence.py           # 출처 신뢰도 평가
│   │   ├── market_context.py       # MarketContext 스키마
│   │   └── state.py                # OrchestratorState 정의
│   ├── tools/
│   │   ├── rag_pipeline.py         # Agentic RAG (FAISS + 쿼리 재작성)
│   │   ├── rag_tool.py             # LangChain @tool 래핑
│   │   └── web_search_tool.py      # Tavily 웹 검색 도구
│   └── logging_utils.py            # 구조화 로깅
├── tests/
│   ├── unit/                       # 모듈별 단위 테스트
│   │   ├── schemas/
│   │   ├── prompts/
│   │   └── tools/
│   └── integration/                # 그래프 전체 흐름 테스트
│       └── test_orchestrator.py
├── app.py                          # 실행 진입점
├── pyproject.toml
└── README.md
```

## Getting Started

```bash
# 의존성 설치
pip install -e .

# 환경변수 설정
cp .env.example .env
# .env에 OPENAI_API_KEY, TAVILY_API_KEY, FAISS 인덱스 경로 입력

# 실행
python app.py

# 테스트
pip install pytest pytest-asyncio
pytest tests/ -v
```

## Contributors

- **김지온** : Comparative SWOT Agent, Report Agent, 보고서 목차 설계
- **서제임스** : Central Orchestrator, Strategy Agent, RAG Pipeline, 테스트
- **이동민** : Market Agent, Agentic RAG 서브그래프, FAISS 인덱스 구축
- **장아현** : Human Review HITL 설계, 프롬프트 엔지니어링, 신뢰도 평가 로직