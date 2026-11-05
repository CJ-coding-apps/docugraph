"""Fixtures shared by the browser-backed integration tests."""

import socket

import pytest


@pytest.fixture
def loopback_alias():
    """Skip when a second loopback address cannot be bound.

    The blocked-host tests need a loopback address other than 127.0.0.1 so the
    policy can allow one host and refuse another. macOS boots ``lo0`` with
    ``127.0.0.1`` only, where binding ``127.0.0.2`` raises ``OSError: [Errno 49]
    Can't assign requested address``. That is an environment limit, not a
    policy regression, so the row skips with this reason instead of failing.

    Returns a callable: pass each address the test is about to bind.
    """

    def require(address: str) -> None:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind((address, 0))
        except OSError as error:
            pytest.skip(
                f"{address} cannot be bound on this host ({error}); the blocked "
                "host needs a second loopback address"
            )

    return require
