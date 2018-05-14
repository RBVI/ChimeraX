# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

from chimerax.core.toolshed import BundleAPI


class _MyAPI(BundleAPI):

    api_version = 1

    @staticmethod
    def register_command(bi, ci, logger):
        from . import cmd
        from chimerax.core.commands import register
        func_attr = ci.name.replace(' ', '_')
        func = getattr(cmd, func_attr)
        desc_attr = func_attr + "_desc"
        desc = getattr(cmd, desc_attr)
        if desc.synopsis is None:
            desc.synopsis = ci.synopsis
        register(ci.name, desc, func)

    @staticmethod
    def start_tool(session, bi, ti):
        from .gui import RegistrationUI
        return RegistrationUI(session, ti)

    @staticmethod
    def initialize(session, bi):
        from .nag import nag
        nag(session)

    @staticmethod
    def finish(session, bi):
        pass

bundle_api = _MyAPI()
