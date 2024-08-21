def ensure_chimerax_initialized():
    import chimerax

    if not getattr(chimerax, "app_bin_dir", None):
        import chimerax.core.__main__

        chimerax.core.__main__.init(["dummy", "--nogui", "--safemode", "--exit"])


def initialize_test_session():
    from chimerax.core.session import Session
    from chimerax.atomic import initialize_atomic
    from chimerax.dist_monitor import _DistMonitorBundleAPI

    session = Session("cx standalone", minimal=True)
    _DistMonitorBundleAPI.initialize(session)
    initialize_atomic(session)
    return session
