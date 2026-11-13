"""Tests for the MetaMo CI prerequisite guard."""

from tests.conftest import _get_metamo_guard_errors


def test_guard_accepts_available_metamo_and_checkpoints(tmp_path):
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "pilot_ppo.pt").touch()
    (tmp_path / "models" / "lunar_mdn.pth").touch()

    errors = _get_metamo_guard_errors(tmp_path, metamo_available=True)

    assert errors == []


def test_guard_reports_unavailable_metamo(tmp_path):
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "pilot_ppo.pt").touch()
    (tmp_path / "models" / "lunar_mdn.pth").touch()

    errors = _get_metamo_guard_errors(tmp_path, metamo_available=False)

    assert errors == ["MetaMo checkout is unavailable."]


def test_guard_reports_missing_checkpoints(tmp_path):
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "pilot_ppo.pt").touch()

    errors = _get_metamo_guard_errors(tmp_path, metamo_available=True)

    assert len(errors) == 1
    assert "models/lunar_mdn.pth" in errors[0]


def test_guard_reports_all_missing_prerequisites(tmp_path):
    errors = _get_metamo_guard_errors(tmp_path, metamo_available=False)

    assert len(errors) == 2
    assert "MetaMo checkout is unavailable." in errors[0]
    assert "models/pilot_ppo.pt" in errors[1]
    assert "models/lunar_mdn.pth" in errors[1]
