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

class OpenerInfo:
    def open(self, session, data, file_name, **kw):
        raise NotImplementedError("Opener did not implement mandatory 'open' method")

    @property
    def open_args(self):
        return {}

class FetcherInfo:
    def fetch(self, session, ident, format_name, ignore_cache, **kw):
        raise NotImplementedError("Fetcher did not implement mandatory 'fetch' method")

    @property
    def fetch_args(self):
        return {}

from .manager import NoOpenerError

from chimerax.core.toolshed import BundleAPI

class _OpenBundleAPI(BundleAPI):

    @staticmethod
    def init_manager(session, bundle_info, name, **kw):
        """Initialize open-command manager"""
        if name == "open command":
            from . import manager
            session.open_command = manager.OpenManager(session)
            return session.open_command

    @staticmethod
    def register_command(command_name, logger):
        from . import cmd
        cmd.register_command(command_name, logger)

bundle_api = _OpenBundleAPI()
