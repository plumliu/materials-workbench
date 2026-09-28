import socket

import pytest


@pytest.fixture(autouse=True, scope="function")
def no_network():
    def blocked(*args, **kwargs):
        raise AssertionError("Tests must not connect to any API or network service")

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(socket.socket, "connect", blocked)
        monkeypatch.setattr(socket, "create_connection", blocked)
        yield
