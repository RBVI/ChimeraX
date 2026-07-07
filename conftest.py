import warnings

import pytest


def _ensure_ssl_cert_file():
    """Point OpenSSL at certifi's CA bundle for the whole test session.

    chimerax.core.__main__.init() does this at app startup, but tests run the
    bare interpreter (make pytest-both-exes / pytest-app / pytest-wheel) and make
    HTTPS requests without ever calling init().  The build also exports a
    SSL_CERT_FILE (mk/config.make) that only resolves inside the installed app,
    and python-build-standalone's bundled OpenSSL has no usable default CA path,
    so without this OpenSSL loads zero CAs and every certificate verification
    fails.  Assign unconditionally, exactly as init() does, so we override that
    stale export rather than deferring to it.
    """
    import os

    try:
        import certifi
    except ImportError:
        return
    os.environ["SSL_CERT_FILE"] = certifi.where()


# Run at conftest import time, before any test is collected or run, so every
# invocation of the test harness has working HTTPS regardless of entry point.
_ensure_ssl_cert_file()


@pytest.fixture(scope="function")
def ensure_chimerax_initialized():
    _ensure_chimerax_initialized()


# You can import this hidden one if you need to run code that needs ChimeraX initialized before
# the tests even run, as in amber_info, which tries to access chimerax.app_bin_dir when you import
# it.
def _ensure_chimerax_initialized():
    import chimerax

    if not getattr(chimerax, "app_bin_dir", None):
        import chimerax.core.__main__

        chimerax.core.__main__.init(["dummy", "--nogui", "--safemode", "--exit"])


_test_session = None


def get_test_session():
    global _test_session
    if _test_session is None:
        from chimerax.core import get_minimal_test_session

        with warnings.catch_warnings(action="ignore"):
            _test_session = get_minimal_test_session()
    return _test_session


@pytest.fixture(scope="function")
def test_production_session():
    session = get_test_session()
    yield session
    session.reset()


def pytest_configure(config):
    markexpr = config.getoption("markexpr")
    if "not wheel" in markexpr:
        # Initialize the test session before tests are even collected, because
        # pytest's usual schtick of importing modules BEFORE the tests are collected
        # totally breaks code that modifies __all__s at runtime. We need ChimeraX to
        # always be the first thing that runs in any tool.
        _ = get_test_session()
