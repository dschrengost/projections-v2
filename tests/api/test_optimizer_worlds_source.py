from __future__ import annotations

from projections.api.optimizer_api import QuickBuildRequest


def test_quick_build_request_defaults_to_gtv2_worlds() -> None:
    request = QuickBuildRequest(date="2026-03-01", draft_group_id=123456)

    assert request.worlds_source == "gtv2"
