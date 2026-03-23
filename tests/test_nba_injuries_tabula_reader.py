from __future__ import annotations

from io import BytesIO

import pandas as pd
from PyPDF2 import PdfWriter

from scrapers.nba_injuries import LAYOUTS, TabulaTableReader


def _build_pdf_bytes(page_count: int) -> bytes:
    writer = PdfWriter()
    for _ in range(page_count):
        writer.add_blank_page(width=72, height=72)
    payload = BytesIO()
    writer.write(payload)
    return payload.getvalue()


def test_tabula_reader_forces_subprocess_mode(monkeypatch) -> None:
    calls: list[dict] = []

    def _fake_read_pdf(*args, **kwargs):  # noqa: ANN002, ANN003
        calls.append(kwargs)
        return [pd.DataFrame({"a": [1]})]

    monkeypatch.setattr("scrapers.nba_injuries.read_pdf", _fake_read_pdf)

    reader = TabulaTableReader()
    out = reader.extract(_build_pdf_bytes(2), LAYOUTS[-1])

    assert len(out) == 2
    assert len(calls) == 2
    assert all(call.get("force_subprocess") is True for call in calls)
