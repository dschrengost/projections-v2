from projections.api.entry_manager_api import _assign_lineup_to_slots_with_maps


def _draftable_maps():
    draftable_ids_by_player = {
        1001: {"PG": 11},
        1002: {"SG": 12},
        1003: {"SF": 13},
        1004: {"PF": 14},
        1005: {"C": 15},
        1006: {"G": 16},
        1007: {"F": 17},
        1008: {"UTIL": 18},
    }
    dk_names_by_player = {
        1001: "PG One",
        1002: "SG Two",
        1003: "SF Three",
        1004: "PF Four",
        1005: "C Five",
        1006: "G Six",
        1007: "F Seven",
        1008: "UTIL Eight",
    }
    return draftable_ids_by_player, dk_names_by_player


def test_assign_lineup_to_slots_accepts_direct_dk_player_ids() -> None:
    draftable_ids_by_player, dk_names_by_player = _draftable_maps()
    internal_to_dk_player_id = {
        "1": 1001,
        "2": 1002,
        "3": 1003,
        "4": 1004,
        "5": 1005,
        "6": 1006,
        "7": 1007,
    }
    internal_to_name = {
        "1": "PG One",
        "2": "SG Two",
        "3": "SF Three",
        "4": "PF Four",
        "5": "C Five",
        "6": "G Six",
        "7": "F Seven",
    }

    slot_values = _assign_lineup_to_slots_with_maps(
        ["1", "2", "3", "4", "5", "6", "7", "1008"],
        internal_to_dk_player_id,
        internal_to_name,
        draftable_ids_by_player,
        dk_names_by_player,
    )

    assert slot_values == {
        "PG": "PG One (11)",
        "SG": "SG Two (12)",
        "SF": "SF Three (13)",
        "PF": "PF Four (14)",
        "C": "C Five (15)",
        "G": "G Six (16)",
        "F": "F Seven (17)",
        "UTIL": "UTIL Eight (18)",
    }


def test_assign_lineup_to_slots_accepts_draftable_ids() -> None:
    draftable_ids_by_player, dk_names_by_player = _draftable_maps()
    internal_to_dk_player_id = {
        "1": 1001,
        "2": 1002,
        "3": 1003,
        "4": 1004,
        "5": 1005,
        "6": 1006,
        "7": 1007,
    }
    internal_to_name = {
        "1": "PG One",
        "2": "SG Two",
        "3": "SF Three",
        "4": "PF Four",
        "5": "C Five",
        "6": "G Six",
        "7": "F Seven",
    }

    slot_values = _assign_lineup_to_slots_with_maps(
        ["1", "2", "3", "4", "5", "6", "7", "18"],
        internal_to_dk_player_id,
        internal_to_name,
        draftable_ids_by_player,
        dk_names_by_player,
    )

    assert slot_values["UTIL"] == "UTIL Eight (18)"
    assert all(slot_values[slot] for slot in slot_values)
