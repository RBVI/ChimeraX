import pytest


@pytest.fixture()
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


@pytest.fixture()
def test_session(ensure_chimerax_initialized):
    from chimerax.core.session import Session
    from chimerax.atomic import initialize_atomic
    from chimerax.dist_monitor import _DistMonitorBundleAPI

    session = Session("cx standalone", minimal=True)
    _DistMonitorBundleAPI.initialize(session)
    initialize_atomic(session)
    return session
