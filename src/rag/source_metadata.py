"""Source metadata helpers for references and citations."""

from __future__ import annotations


SOURCE_METADATA = {
    "market_report.pdf": {
        "publisher": "IEA",
        "title": "Global EV Outlook 2025",
        "url": "https://www.iea.org/reports/global-ev-outlook-2025",
        "reference_text": "IEA(2025). *Global EV Outlook 2025*. IEA. https://www.iea.org/reports/global-ev-outlook-2025",
    },
    "analyst_report.pdf": {
        "publisher": "유진투자증권",
        "title": "K배터리 리포트 추출본",
        "url": "https://www.eugenefn.com/",
        "reference_text": "유진투자증권(2025). *K배터리 리포트 추출본*. 유진투자증권 리서치. https://www.eugenefn.com/",
    },
    "catl.pdf": {
        "publisher": "CATL",
        "title": "2025 Annual Report",
        "url": "https://www.catl.com/en/news/6773.html",
        "reference_text": "CATL(2026). *2025 Annual Report*. CATL. https://www.catl.com/en/news/6773.html",
    },
    "skon.pdf": {
        "publisher": "SK이노베이션",
        "title": "사업보고서",
        "url": "https://dart.fss.or.kr/navi/searchNavi.do?naviCode=A002&naviCrpCik=00631518&naviCrpNm=SK%EC%9D%B4%EB%85%B8%EB%B2%A0%EC%9D%B4%EC%85%98",
        "reference_text": "SK이노베이션(2025). *사업보고서*. DART. https://dart.fss.or.kr/navi/searchNavi.do?naviCode=A002&naviCrpCik=00631518&naviCrpNm=SK%EC%9D%B4%EB%85%B8%EB%B2%A0%EC%9D%B4%EC%85%98",
    },
}

DOCUMENT_ID_TO_SOURCE = {
    "market-report": "market_report.pdf",
    "analyst-report": "analyst_report.pdf",
    "catl": "catl.pdf",
    "skon": "skon.pdf",
}


def infer_source_name(*, source_id: str = "", source: str = "", title: str = "") -> str:
    """Infer the canonical source filename from available identifiers."""

    if source in SOURCE_METADATA:
        return source
    if title in SOURCE_METADATA:
        return title

    normalized_title = title.strip().lower()
    if normalized_title.endswith(".pdf"):
        for source_name in SOURCE_METADATA:
            if source_name.lower() == normalized_title:
                return source_name

    if source_id:
        normalized_id = source_id
        if normalized_id.startswith("rag_"):
            normalized_id = normalized_id[4:]
        for doc_id, source_name in DOCUMENT_ID_TO_SOURCE.items():
            if normalized_id.startswith(doc_id):
                return source_name

    return source or title


def resolve_source_metadata(*, source_id: str = "", source: str = "", title: str = "") -> dict[str, str]:
    """Resolve source metadata from filename-like identifiers."""

    source_name = infer_source_name(source_id=source_id, source=source, title=title)
    metadata = SOURCE_METADATA.get(source_name, {})
    return {
        "source_name": source_name,
        "title": metadata.get("title", title or source_name),
        "url": metadata.get("url", ""),
        "reference_text": metadata.get("reference_text", ""),
        "publisher": metadata.get("publisher", ""),
    }
