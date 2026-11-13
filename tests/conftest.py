"""Pytest configuration for repository-wide test requirements."""

import os
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent

REQUIRED_CHECKPOINTS = (
    Path("models/pilot_ppo.pt"),
    Path("models/lunar_mdn.pth"),
)


def _get_metamo_guard_errors(repo_root, metamo_available):
    """Return any missing prerequisites for MetaMo-required test runs."""
    errors = []

    if not metamo_available:
        errors.append("MetaMo checkout is unavailable.")

    missing_checkpoints = [
        str(checkpoint)
        for checkpoint in REQUIRED_CHECKPOINTS
        if not (repo_root / checkpoint).is_file()
    ]

    if missing_checkpoints:
        errors.append(
            "Required checkpoint(s) missing: "
            + ", ".join(missing_checkpoints)
        )

    return errors


def pytest_collection_modifyitems(session, config, items):
    """Fail test collection when CI requires MetaMo but prerequisites are absent."""
    if os.environ.get("SUBREP_REQUIRE_METAMO") != "1":
        return

    from bridge._loader import is_available

    errors = _get_metamo_guard_errors(REPO_ROOT, is_available())

    if errors:
        message = (
            "SUBREP_REQUIRE_METAMO=1, but required MetaMo prerequisites "
            "are not available:\n  - "
            + "\n  - ".join(errors)
        )
        raise pytest.UsageError(message)
