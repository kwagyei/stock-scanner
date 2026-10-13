"""The shared guard in conftest.py: every test runs offline, without having to ask for it."""

import urllib.request

import pytest
import yfinance

import scanner


def test_python_sockets_are_blocked():
    with pytest.raises(RuntimeError, match="network access in tests"):
        urllib.request.urlopen("https://example.com", timeout=1)


def test_yfinance_is_blocked_even_though_it_bypasses_python_sockets():
    with pytest.raises(RuntimeError, match="network access in tests"):
        yfinance.Ticker("ELTEL.ST")
    # an unmocked price fetch in the scanner fails safely (warns, returns nothing)
    assert scanner.fetch_history("ELTEL.ST", "1mo").empty


def test_the_guard_loads_without_yfinance():
    """The fund-watch job runs these tests with only feedparser, PyYAML, certifi and pytest
    installed; the shared conftest must not need yfinance."""
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    code = "import sys; sys.modules['yfinance'] = None; import conftest; print(conftest.yfinance)"
    out = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "None"
