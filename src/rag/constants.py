from __future__ import annotations

"""Shared retrieval constants."""

# 문서 출처 그룹과 에이전트별 기본 매핑 규칙입니다.
MARKET_SOURCES = {"market_report.pdf", "analyst_report.pdf"}

COMPANY_SOURCE_BY_COLLECTION = {
    "skon_agent": "skon.pdf",
    "catl_agent": "catl.pdf",
}

# 질문 확장에 쓰는 힌트들입니다.
ENTITY_HINTS = {
    "market_agent": ["global EV market", "battery market outlook", "전기차 시장", "배터리 시장"],
    "skon_agent": ["SK On", "SK온", "SK이노베이션 배터리"],
    "catl_agent": ["CATL", "宁德时代", "Contemporary Amperex Technology"],
    "swot_agent": ["SK On", "CATL", "comparative SWOT", "비교 SWOT"],
    "report_agent": ["battery strategy report", "market and company evidence", "전략 보고서 근거"],
}

# 질문 주제별로 추가 검색어를 붙일 때 사용합니다.
TOPIC_HINTS = {
    "생산": ["생산능력", "가동률", "생산실적", "공장", "capacity", "utilization"],
    "해외": ["해외 법인", "JV", "미국", "유럽", "헝가리", "중국", "overseas entity"],
    "재무": ["매출", "영업이익", "순이익", "분기 실적", "financial performance", "quarterly revenue"],
    "전략": ["전략", "투자", "고객", "수주", "competitive advantage", "strategy"],
    "시장": ["시장 전망", "수요", "정책", "EV adoption", "market outlook"],
    "비교": ["comparative SWOT", "강점", "약점", "opportunity", "threat"],
}

# 검색 점수와 coverage 판단에 쓰는 키워드입니다.
NUMERIC_HINTS = ("매출", "실적", "생산", "가동률", "재무", "capacity", "revenue", "profit", "quarter")
STRATEGY_HINTS = ("전략", "시장", "경쟁", "risk", "리스크", "opportunity", "threat", "SWOT")

# 키워드 추출 시 제외할 일반 단어 목록입니다.
STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "into",
    "about",
    "what",
    "which",
    "when",
    "where",
    "how",
    "global",
    "market",
    "agent",
    "report",
    "자료",
    "보고서",
    "관련",
    "대한",
    "무엇",
    "어떻게",
    "정리",
    "설명",
    "알려줘",
    "현황",
}

# evidence packet에 topic tag를 붙일 때 사용합니다.
TOPIC_KEYWORDS = {
    "market": ("시장", "수요", "정책", "outlook", "demand", "market", "ev", "ess"),
    "finance": ("매출", "영업이익", "순이익", "재무", "revenue", "profit", "financial", "quarter"),
    "capacity": ("생산능력", "생산", "가동률", "capacity", "utilization", "plant", "factory"),
    "overseas": ("해외", "법인", "미국", "유럽", "중국", "헝가리", "jv", "entity", "subsidiary"),
    "strategy": ("전략", "투자", "고객", "수주", "경쟁", "strategy", "customer", "investment"),
    "swot": ("swot", "강점", "약점", "opportunity", "threat", "risk", "리스크"),
}
