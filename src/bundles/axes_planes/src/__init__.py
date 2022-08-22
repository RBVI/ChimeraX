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

from .cmd import AxisModelArg, AxisModelsArg, PlaneModelArg, PlaneModelsArg, PlaneModel, AxisModel

class AxesPlanes_API(BundleAPI):

    @staticmethod
    def get_class(class_name):
        if class_name == "PlaneModel":
            from .cmd import PlaneModel
            return PlaneModel
        elif class_name == "AxisModel":
            from .cmd import AxisModel
            return AxisModel

    @staticmethod
    def register_command(command_name, logger):
        from . import cmd
        cmd.register_command(command_name, logger)

bundle_api = AxesPlanes_API()
