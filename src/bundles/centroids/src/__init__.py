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

from .centroid import centroid
from .cmd import CentroidModel

from chimerax.core.toolshed import BundleAPI

class CentroidsAPI(BundleAPI):

    @staticmethod
    def get_class(class_name):
        if class_name == "CentroidModel":
            return CentroidModel
        raise ValueError("Don't know about class %s" % class_name)

    @staticmethod
    def register_command(command_name, logger):
        from . import cmd
        cmd.register_command(command_name, logger)

bundle_api = CentroidsAPI()
