"""Shared test setup: no test may touch the network.

Python sockets are blocked for every test, which covers urllib, feedparser and the Anthropic
SDK. yfinance downloads through curl_cffi (a C library that bypasses Python sockets), so it is
blocked separately; tests that need prices mock the fetch functions or yfinance.Ticker.
"""

import socket

import pytest

try:  # the fund-watch job installs only feedparser, PyYAML, certifi and pytest
    import yfinance
except ImportError:
    yfinance = None


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def guard(*args, **kwargs):
        raise RuntimeError("network access in tests")

    monkeypatch.setattr(socket.socket, "connect", guard)
    monkeypatch.setattr(socket, "create_connection", guard)
    if yfinance is not None:
        monkeypatch.setattr(yfinance, "Ticker", guard)
        monkeypatch.setattr(yfinance, "download", guard)
