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
        # 'register_command' is lazily called when the command is referenced
        from . import test
        from chimerax.core.commands import register, CmdDesc, BoolArg, RestOfLine
        if ci.name == "debug test":
            desc = CmdDesc(synopsis=ci.synopsis,
                           keyword=[('stderr', BoolArg)])
            register(ci.name, desc, test.run_commands, logger=logger)
        elif ci.name == "debug exectest":
            desc = CmdDesc(synopsis=ci.synopsis)

            def func(session, bi=bi):
                test.run_exectest(session, bi)
            register(ci.name, desc, func, logger=logger)
        elif ci.name == "debug expectfail":
            desc = CmdDesc(synopsis=ci.synopsis, required=[('command', RestOfLine)])
            register(ci.name, desc, test.run_expectfail, logger=logger)


bundle_api = _MyAPI()
