from projections.optimizer.model_spec import SpecPlayer
from projections.optimizer.quick_build import _is_assignable_dk_lineup


def test_is_assignable_dk_lineup_rejects_unassignable_combo() -> None:
    # Mirrors a real failing lineup shape (no feasible way to fill both C and F).
    pid_to_player = {
        "1627759": SpecPlayer("1627759", "Jaylen Brown", "BOS", ["SF", "SG"], 8000, 40.0),
        "1629656": SpecPlayer("1629656", "Quentin Grimes", "PHI", ["PG", "SG"], 6000, 30.0),
        "1630178": SpecPlayer("1630178", "Tyrese Maxey", "PHI", ["PG"], 9000, 45.0),
        "1631230": SpecPlayer("1631230", "Dominick Barlow", "PHI", ["C", "PF"], 4500, 20.0),
        "1642285": SpecPlayer("1642285", "Cam Spencer", "MEM", ["PG", "SG"], 4300, 18.0),
        "1642383": SpecPlayer("1642383", "Walter Clayton Jr.", "UTA", ["SG"], 3900, 16.0),
        "1642883": SpecPlayer("1642883", "Sion James", "CHA", ["PF", "SF"], 4100, 17.0),
        "201950": SpecPlayer("201950", "Jrue Holiday", "BOS", ["PG"], 7000, 33.0),
    }
    lineup = (
        "1627759",
        "1629656",
        "1630178",
        "1631230",
        "1642285",
        "1642383",
        "1642883",
        "201950",
    )
    assert _is_assignable_dk_lineup(lineup, pid_to_player, lineup_size=8) is False


def test_is_assignable_dk_lineup_accepts_valid_combo() -> None:
    pid_to_player = {
        "pg1": SpecPlayer("pg1", "PG One", "A", ["PG"], 5000, 25.0),
        "sg1": SpecPlayer("sg1", "SG One", "A", ["SG"], 5000, 24.0),
        "sf1": SpecPlayer("sf1", "SF One", "A", ["SF"], 5000, 23.0),
        "pf1": SpecPlayer("pf1", "PF One", "A", ["PF"], 5000, 22.0),
        "c1": SpecPlayer("c1", "C One", "A", ["C"], 5000, 21.0),
        "g1": SpecPlayer("g1", "G One", "A", ["PG", "SG"], 5000, 20.0),
        "f1": SpecPlayer("f1", "F One", "A", ["SF", "PF"], 5000, 19.0),
        "u1": SpecPlayer("u1", "U One", "A", ["SG"], 5000, 18.0),
    }
    lineup = ("pg1", "sg1", "sf1", "pf1", "c1", "g1", "f1", "u1")
    assert _is_assignable_dk_lineup(lineup, pid_to_player, lineup_size=8) is True
