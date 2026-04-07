"""
Market Agent 출력 스키마 및 Mock 데이터.
Phase 1과의 인터페이스 계약서 역할.
"""

from typing import TypedDict


# ─────────────────────────────────────────────
# MarketContext 스키마
# ─────────────────────────────────────────────

class MarketContext(TypedDict):
    ev_growth_slowdown: dict    # 캐즘 현황 (성장률 둔화 수치, 지역별)
    market_share_ranking: dict  # 글로벌 배터리 점유율 순위
    lfp_ncm_trend: dict         # LFP vs NCM 기술 트렌드
    ess_hev_growth: dict        # ESS/HEV 시장 성장성 수치
    regulatory_status: dict     # IRA/관세/EU 규제 현황
    cost_competitiveness: dict  # 원가 경쟁력 트렌드


# ─────────────────────────────────────────────
# Mock 데이터
# ─────────────────────────────────────────────

MOCK_MARKET_CONTEXT: MarketContext = {
    "ev_growth_slowdown": {
        "global_growth_rate": {
            "2022": 0.68,
            "2023": 0.31,
            "2024_estimate": 0.18,
        },
        "regional_breakdown": {
            "china":         {"2023": 0.36, "2024_estimate": 0.20},
            "europe":        {"2023": 0.14, "2024_estimate": 0.06},
            "north_america": {"2023": 0.40, "2024_estimate": 0.09},
        },
        "source": "BloombergNEF EV Outlook 2024",
        "key_narrative": (
            "2023년부터 EV 성장률 급격히 둔화, "
            "특히 유럽/북미 시장에서 캐즘 현상 뚜렷"
        ),
    },

    "market_share_ranking": {
        "year": "2024_H1",
        "rankings": [
            {"rank": 1, "company": "CATL",     "share": 0.368},
            {"rank": 2, "company": "BYD",      "share": 0.157},
            {"rank": 3, "company": "LGES",     "share": 0.131},
            {"rank": 4, "company": "Panasonic","share": 0.076},
            {"rank": 5, "company": "SDI",      "share": 0.054},
            {"rank": 6, "company": "SKON",     "share": 0.049},
        ],
        "source": "SNE Research 2024 H1",
        "key_narrative": (
            "CATL 압도적 1위 유지, SK On은 6위권으로 하락세"
        ),
    },

    "lfp_ncm_trend": {
        "lfp_share_trend": {
            "2021": 0.41,
            "2022": 0.52,
            "2023": 0.58,
            "2024_estimate": 0.62,
        },
        "ncm_share_trend": {
            "2021": 0.59,
            "2022": 0.48,
            "2023": 0.42,
            "2024_estimate": 0.38,
        },
        "regional_preference": {
            "china":         "LFP 우세 (75% 이상)",
            "europe":        "NCM 우세 (주행거리 중시)",
            "north_america": "NCM 우세 (테슬라 제외)",
        },
        "source": "Wood Mackenzie Battery Technology Report 2024",
        "key_narrative": (
            "LFP 점유율 지속 확대, 원가 우위로 중저가 세그먼트 장악 중"
        ),
    },

    "ess_hev_growth": {
        "ess": {
            "global_market_size_gwh": {"2023": 185, "2025_forecast": 420},
            "cagr_2023_2028": 0.31,
            "key_drivers": ["재생에너지 확대", "전력망 안정화 수요", "기업 RE100"],
            "source": "BloombergNEF Energy Storage Outlook 2024",
            "key_narrative": "EV 캐즘과 무관하게 ESS는 고성장 지속",
        },
        "hev": {
            "growth_rate_2024": 0.22,
            "source": "IEA Global EV Outlook 2024",
            "key_narrative": "완전 EV 전환 지연으로 HEV 수요 반사이익",
        },
    },

    "regulatory_status": {
        "ira": {
            "status": "시행 중",
            "key_provisions": [
                "북미산 배터리 세액공제 최대 $45/kWh",
                "중국산 배터리 부품 사용 제한 (2024년부터 단계적)",
            ],
            "impact_on_chinese_makers": "CATL 등 중국 업체 직접 판매 제한",
            "source": "U.S. Department of Energy IRA Guidance 2024",
            "key_narrative": (
                "IRA는 북미 생산거점 보유 업체에 유리, CATL에 직접적 장벽"
            ),
        },
        "us_tariffs": {
            "chinese_ev_battery_tariff_2024": 0.25,
            "chinese_ev_battery_tariff_2026": 0.50,
            "source": "USTR Section 301 Tariff Review 2024",
            "key_narrative": "관세 단계적 인상으로 CATL 우회 전략 필요성 증대",
        },
        "eu_regulation": {
            "battery_passport_mandatory": "2027",
            "carbon_footprint_disclosure": "2025년부터",
            "source": "EU Battery Regulation (EU) 2023/1542",
            "key_narrative": (
                "EU 배터리법은 공급망 추적 가능성 요구, 준비된 업체에 유리"
            ),
        },
    },

    "cost_competitiveness": {
        "average_pack_cost_usd_per_kwh": {
            "2022": 151,
            "2023": 139,
            "2024_estimate": 115,
        },
        "by_chemistry": {
            "lfp": {"2024_estimate": 90},
            "ncm": {"2024_estimate": 130},
        },
        "source": "BloombergNEF Battery Price Survey 2024",
        "key_narrative": (
            "LFP 원가가 NCM 대비 30% 낮아 CATL의 원가 우위 구조적으로 지속"
        ),
    },
}
