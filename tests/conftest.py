import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import update_data as u  # noqa: E402


@pytest.fixture
def prices_dir(tmp_path, monkeypatch):
    """Redirects the module's price-file directory to an empty temp dir, so
    tests never touch the real data/prices/ and don't see each other's files."""
    d = tmp_path / "prices"
    monkeypatch.setattr(u, "PRICES", d)
    return d
