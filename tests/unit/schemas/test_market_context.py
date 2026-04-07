"""
schemas/market_context.py 단위 테스트.

검증 항목:
  - MOCK_MARKET_CONTEXT가 MarketContext의 6개 축 필드를 모두 포함
  - 각 축에 source 및 key_narrative 필드 존재
  - 수치 데이터 타입 정확성
"""

import pytest
from schemas.market_context import MarketContext, MOCK_MARKET_CONTEXT


REQUIRED_AXES = [
    "ev_growth_slowdown",
    "market_share_ranking",
    "lfp_ncm_trend",
    "ess_hev_growth",
    "regulatory_status",
    "cost_competitiveness",
]


class TestMarketContextSchema:

    def test_market_context_has_all_axes(self):
        fields = MarketContext.__annotations__
        for axis in REQUIRED_AXES:
            assert axis in fields

    def test_market_context_all_axes_are_dict(self):
        for axis in REQUIRED_AXES:
            assert MarketContext.__annotations__[axis] == dict


class TestMockMarketContext:

    def test_mock_has_all_required_axes(self):
        for axis in REQUIRED_AXES:
            assert axis in MOCK_MARKET_CONTEXT, f"'{axis}' 누락"

    def test_each_axis_has_source(self):
        """출처 추적 가능성 — 각 축 또는 하위 섹션에 source 필드 필수"""
        def _has_source(data: dict) -> bool:
            if "source" in data:
                return True
            return any(_has_source(v) for v in data.values() if isinstance(v, dict))

        for axis in REQUIRED_AXES:
            data = MOCK_MARKET_CONTEXT[axis]
            assert _has_source(data), f"'{axis}'에 source 없음 (하위 포함)"

    def test_each_axis_has_key_narrative(self):
        """시장 서사 — 각 축 또는 하위 섹션에 key_narrative 필드 필수"""
        def _has_narrative(data: dict) -> bool:
            if "key_narrative" in data:
                return True
            return any(_has_narrative(v) for v in data.values() if isinstance(v, dict))

        for axis in REQUIRED_AXES:
            data = MOCK_MARKET_CONTEXT[axis]
            assert _has_narrative(data), f"'{axis}'에 key_narrative 없음 (하위 포함)"

    def test_ev_growth_slowdown_structure(self):
        ev = MOCK_MARKET_CONTEXT["ev_growth_slowdown"]
        assert "global_growth_rate" in ev
        assert "regional_breakdown" in ev
        assert isinstance(ev["global_growth_rate"], dict)
        # 성장률은 0~1 사이 float
        for year, rate in ev["global_growth_rate"].items():
            assert 0 <= rate <= 1, f"{year} 성장률이 범위를 벗어남: {rate}"

    def test_market_share_ranking_structure(self):
        ms = MOCK_MARKET_CONTEXT["market_share_ranking"]
        assert "rankings" in ms
        assert isinstance(ms["rankings"], list)
        assert len(ms["rankings"]) > 0
        # 점유율 합계가 1 이하
        total_share = sum(r["share"] for r in ms["rankings"])
        assert total_share <= 1.01  # 부동소수점 허용 오차

    def test_market_share_includes_skon_and_catl(self):
        rankings = MOCK_MARKET_CONTEXT["market_share_ranking"]["rankings"]
        companies = [r["company"] for r in rankings]
        assert "SKON" in companies
        assert "CATL" in companies

    def test_cost_competitiveness_lfp_cheaper_than_ncm(self):
        """LFP가 NCM보다 저렴해야 함 — 설계 전제 조건"""
        by_chem = MOCK_MARKET_CONTEXT["cost_competitiveness"]["by_chemistry"]
        lfp_cost = by_chem["lfp"]["2024_estimate"]
        ncm_cost = by_chem["ncm"]["2024_estimate"]
        assert lfp_cost < ncm_cost
