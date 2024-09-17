import pytest


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
        from chimerax.core.__main__ import _set_app_dirs
        from chimerax.core import version
        from chimerax.core.session import Session, register_misc_commands
        from chimerax.core import core_settings
        from chimerax.core import toolshed
        from chimerax.atomic import initialize_atomic
        from chimerax.dist_monitor import _DistMonitorBundleAPI
        from chimerax.core.session import register_misc_commands

        _set_app_dirs(version)
        session = Session("cx standalone", minimal=False)
        session.ui.is_gui = False
        core_settings.init(session)
        register_misc_commands(session)

        from chimerax.core import attributes

        from chimerax.core.nogui import NoGuiLog

        session.logger.add_log(NoGuiLog())

        attributes.RegAttrManager(session)

        toolshed.init(
            session.logger,
            debug=session.debug,
            check_available=False,
            remote_url=toolshed.default_toolshed_url(),
            session=session,
        )

        session.toolshed = toolshed.get_toolshed()

        session.toolshed.bootstrap_bundles(session, safe_mode=False)
        from chimerax.core import tools

        session.tools = tools.Tools(session, first=True)
        from chimerax.core import undo

        session.undo = undo.Undo(session, first=True)
        _test_session = session
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
