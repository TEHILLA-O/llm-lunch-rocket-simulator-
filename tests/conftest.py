"""Storage isolation for tests. Physics tests never touch the developer's data files."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    monkeypatch.setenv("LUNARMIND_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("LUNARMIND_MODELS", str(tmp_path / "models"))
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
