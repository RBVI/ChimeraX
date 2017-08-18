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

class _BondRotBundleAPI(BundleAPI):

    @staticmethod
    def get_class(class_name):
        if class_name == "BondRotationManager":
            from . import manager
            return manager.BondRotationManager
        elif class_name == "BondRotation":
            from . import bond_rot
            return bond_rot.BondRotation

    @staticmethod
    def initialize(session, bundle_info):
        """Install bond-rotation manager into existing session"""
        from .manager import BondRotationManager
        session.bond_rotations = BondRotationManager(session, bundle_info)

    @staticmethod
    def finish(session, bundle_info):
        """De-install bond-rotation manager from existing session"""
        del session.bond_rotations

    """
    @staticmethod
    def register_command(command_name, logger):
        # 'register_command' is lazily called when the command is referenced
        from . import cmd
        cmd.register_seqalign_command(logger)
    """

bundle_api = _BondRotBundleAPI()
