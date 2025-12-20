from __future__ import annotations

from projections.contest_sim.field_library import FieldLibrary, load_field_library, save_field_library


def test_field_library_roundtrip(tmp_path) -> None:
    lib = FieldLibrary(
        lineups=[["1", "2"], ["3", "4"]],
        weights=[10, 20],
        meta={"method": "test", "generated_at": "2099-01-01T00:00:00Z"},
    )
    path = tmp_path / "field_library_test.json"
    save_field_library(lib, path)
    loaded = load_field_library(path)

    assert loaded.lineups == lib.lineups
    assert loaded.weights == lib.weights
    assert loaded.meta["method"] == "test"

