# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This particular file is part of the ChimeraX library. You can also
# redistribute and/or modify it under the terms of the GNU Lesser General
# Public License version 2.1 as published by the Free Software Foundation.
# For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
# EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
# LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
# VERSION 2.1
#
# This notice must be embedded in or attached to all copies, including partial
# copies, of the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

"""
chimerax.core: collection of base ChimeraX functionality
========================================================

"""

from .buildinfo import version

__version__ = version


def get_minimal_test_session():
    """Experimental API"""
    # Highly experimental API that sets up a minimal test session. This code is
    # here because we cannot ship our internal conftest.py with ChimeraX. Users would
    # have to figure out how to set up their own test sessions, which would inevitably
    # involve copying our conftest.py and never updating it again. So that code lives
    # here now, where everyone can get updates.
    import warnings
    from chimerax.core.__main__ import _set_app_dirs
    from chimerax.core import version
    from chimerax.core.session import Session, register_misc_commands
    from chimerax.core import core_settings
    from chimerax.core import toolshed
    from chimerax.atomic import initialize_atomic
    from chimerax.dist_monitor import _DistMonitorBundleAPI
    from chimerax.core.session import register_misc_commands
    from chimerax.core import nogui

    warnings.warn(
        "This function is not intended for use in the ChimeraX library. It is for testing ChimeraX modules and bundles. If and only if you are, as intended, calling it in your test harness, catch and ignore this warning. We will close without investigation tickets that are reported if this session is being used."
    )

    _set_app_dirs(version)
    session = Session(minimal=False)
    session.ui = nogui.UI(session)
    session.logger.add_log(nogui.NoGuiLog())
    session.ui.initialize_color_output(True)  # Colored text

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
    return session


def runtime_env_is_chimerax_app():
    import chimerax

    return hasattr(chimerax, "app_dirs")


import os


def _path_to_src() -> str:
    return os.path.dirname(__file__)


def get_lib() -> str:
    return os.path.join(_path_to_src(), "lib")


def get_include() -> str:
    return os.path.join(_path_to_src(), "include")


from .toolshed import BundleAPI

BUNDLE_NAME = "ChimeraX-Core"

_class_cache = {}
# list modules classes are found in used by session restore to recreate objects.
_class_class_init = {
    "UserAliases": ".session",
    "CameraClipPlane": "chimerax.graphics",
    "ClipPlane": "chimerax.graphics",
    "Color": ".colors",
    "Colormap": ".colors",
    "Drawing": "chimerax.graphics",
    "Generic3DModel": ".generic3d",
    "Job": ".tasks",
    "Lighting": "chimerax.graphics",
    "Material": "chimerax.graphics",
    "Model": ".models",
    "Models": ".models",
    "MonoCamera": "chimerax.graphics",
    "OrthographicCamera": "chimerax.graphics",
    "Place": "chimerax.geometry",
    "Places": "chimerax.geometry",
    "SceneClipPlane": "chimerax.graphics",
    "Surface": ".models",
    "Tasks": ".tasks",
    "Tools": ".tools",
    "Undo": ".undo",
    "UserColors": ".colors",
    "UserColormaps": ".colors",
    "View": "chimerax.graphics",
    "_Input": ".nogui",
    # atomic classes moved to a separate bundle; listed here
    # for backwards compatibility
    # (perhaps remove at 1.0? No. Removing breaks loading old sessions.)
    "Atom": "chimerax.atomic",
    "AtomicStructure": "chimerax.atomic",
    "AtomicStructures": "chimerax.atomic",
    "Atoms": "chimerax.atomic",
    "Bond": "chimerax.atomic",
    "Bonds": "chimerax.atomic",
    "Chain": "chimerax.atomic",
    "Chains": "chimerax.atomic",
    "CoordSet": "chimerax.atomic",
    "LevelOfDetail": "chimerax.atomic",
    "MolecularSurface": "chimerax.atomic",
    "PseudobondGroup": "chimerax.atomic",
    "PseudobondManager": "chimerax.atomic",
    "Pseudobond": "chimerax.atomic",
    "Pseudobonds": "chimerax.atomic",
    "Residue": "chimerax.atomic",
    "Residues": "chimerax.atomic",
    "SeqMatchMap": "chimerax.atomic",
    "Sequence": "chimerax.atomic",
    "Structure": "chimerax.atomic",
    "StructureSeq": "chimerax.atomic",
    "AttrRegistration": ".attributes",
    "_NoDefault": ".attributes",
    "RegAttrManager": ".attributes",
    "XSectionManager": "chimerax.atomic.ribbon",
    # For a while sessions were saved with these even though
    # we really didn't want them to be restored
    "NewerVersionQuery": ".toolshed",
}


class _MyAPI(BundleAPI):
    @staticmethod
    def get_class(class_name):
        """Return chimerax.core class for instance in a session

        Parameters
        ----------
        class_name : str
            Class name
        """
        try:
            return _class_cache[class_name]
        except KeyError:
            pass
        module_name = _class_class_init.get(class_name, None)
        if module_name is None:
            cls = None
        else:
            import importlib

            mod = importlib.import_module(module_name, __package__)
            cls = getattr(mod, class_name)
        _class_cache[class_name] = cls
        return cls


bundle_api = _MyAPI()


def profile(func):
    def wrapper(*args, **kw):
        import sys
        import cProfile
        import pstats

        prof = cProfile.Profile()
        v = prof.runcall(func, *args, **kw)
        print(func.__name__, file=sys.__stderr__)
        p = pstats.Stats(prof, stream=sys.__stderr__)
        p.strip_dirs().sort_stats("cumulative", "time").print_callers(40)
        return v

    return wrapper


def is_daily_build():
    """Supported API. Return if ChimeraX Core is from a daily build."""
    # Daily builds are development releases
    from packaging.version import Version

    ver = Version(version)
    return ver.is_devrelease
