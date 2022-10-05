import importlib
import logging
import sys

_log = logging.getLogger()

try:
    import chimerax.ui
    _has_ui = True
except (ModuleNotFoundError, ImportError):
    _has_ui = False

checked_modules = {}

def requires_gui(func):
    if not _has_ui:
        def noGuiFound(*args, **kwargs):
            # TODO:
            # 1. Investigate and document why this works and prints text
            #    to the ChimeraX log on screen.
            # 2. Investigate and document whether a file log can be created
            #    directly, for debugging and bundle developer
            # 3. Investigate and document why this works on a class method.
            #    Working theory is that self is simply a special name for the
            #    first argument to a class function? Review python documentation.
            #    Must get wrapped into *args
            # 4. If a clear understanding of what's going on allows, then nuke all
            #    calls to session.logger.info, session.logger.warning that take in
            #    pure strings only with no formatting from orbit.
            # 5. If that's possible, make the logger fully compatible with
            #    python's logging object
            # 6. If that too is possible, investigate what really needs a session,
            #    and what merely wants a logger, and decouple the session file from
            #    it
            # 7. If even that's possible, investigate registering the ChimeraX logger
            #    as one of the logger formatters
            _log.warning("Calling function that requires ChimeraX GUI, but GUI module not found")
        func = noGuiFound
    return func

def requires_module(module: str):
    def decorator(func):
        _have_module = checked_modules.get(module, False)
        if not _have_module:
            try:
                importlib.import_module(module)
                checked_modules[module] = True
                return func
            except ImportError:
                def moduleNotFound(*args, **kwargs):
                    _log.warning("Calling function that requires %s, but %s not found" % (module, module))
                func = moduleNotFound
        return func
    return decorator
