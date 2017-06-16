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

    @staticmethod
    def start_tool(session, tool_name):
        # If providing more than one tool in package,
        # look at the tool name to see which is being started.
        from . import tool
        try:
            ui = getattr(tool, "bogusUI")
        except AttributeError:
            raise RuntimeError("cannot find UI for tool \"%s\"" % tool_name)
        else:
            return ui(session)

    @staticmethod
    def register_command(command_name, logger):
        from . import cmd
        from chimerax.core.commands import register
        desc_suffix = "_desc"
        for attr_name in dir(cmd):
            if not attr_name.endswith(desc_suffix):
                continue
            subcommand_name = attr_name[:-len(desc_suffix)]
            try:
                func = getattr(cmd, subcommand_name)
            except AttributeError:
                print("no function for \"%s\"" % subcommand_name)
                continue
            desc = getattr(cmd, attr_name)
            register(command_name + ' ' + subcommand_name, desc, func, logger=logger)

        from chimerax.core.commands import atomspec
        atomspec.register_selector("odd", _odd_models, logger)

    @staticmethod
    def get_class(class_name):
        # 'get_class' is called by session code to get class saved in a session
        if class_name == 'BogusUI':
            from . import tool
            return tool.BogusUI
        return None

bundle_api = _MyAPI()


def _odd_models(session, models, results):
    for m in models:
        if m.id[0] % 2:
            results.add_model(m)
            results.add_atoms(m.atoms)
