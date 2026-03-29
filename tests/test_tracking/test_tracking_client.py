from __future__ import annotations

from datetime import date

import httpx

from projections.data.nba import tracking_client


def test_fetch_leaguedashptstats_uses_stats_headers() -> None:
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["referer"] = request.headers.get("Referer", "")
        captured["sec_ch_ua"] = request.headers.get("Sec-Ch-Ua", "")
        captured["accept_encoding"] = request.headers.get("Accept-Encoding", "")
        payload = {
            "resultSets": [
                {
                    "name": "LeagueDashPtStats",
                    "headers": ["PLAYER_ID", "MIN"],
                    "rowSet": [[1, 30.0]],
                }
            ]
        }
        return httpx.Response(status_code=200, json=payload)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        payload = tracking_client.fetch_leaguedashptstats(
            season="2025-26",
            season_type="Regular Season",
            pt_measure_type="Possessions",
            game_date=date(2026, 3, 12),
            timeout=5.0,
            max_retries=1,
            retry_delay=0.0,
            client=client,
        )
    finally:
        client.close()

    assert "resultSets" in payload
    assert "PtMeasureType=Possessions" in captured["url"]
    assert "DateFrom=03%2F12%2F2026" in captured["url"]
    assert captured["referer"] == "https://stats.nba.com/"
    assert "Chromium" in captured["sec_ch_ua"]
    assert "gzip" in captured["accept_encoding"]
