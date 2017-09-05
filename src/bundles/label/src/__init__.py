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

class _LabelBundle(BundleAPI):

    @staticmethod
    def register_command(command_name, logger):
        # 'register_command' is lazily called when the command is referenced
        from . import label2d, label3d
        label2d.register_label_command(logger)
        label3d.register_label_command(logger)

    @staticmethod
    def get_class(class_name):
        # 'get_class' is called by session code to get class saved in a session
        if class_name == 'ObjectLabels':
            from .label3d import ObjectLabels
            return ObjectLabels
        elif class_name == 'Labels':
            from .label2d import Labels
            return Labels
        return None

bundle_api = _LabelBundle()

from .label2d import label_create, label_change, label_delete, Label
