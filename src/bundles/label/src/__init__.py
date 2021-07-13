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
    def initialize(session, bundle_info):
        """Register label mouse modes."""
        if session.ui.is_gui:
            from . import mouselabel, movelabel
            mouselabel.register_mousemode(session)
            movelabel.register_mousemode(session)

        # Register preferences
        from . import settings
        settings.settings = settings._LabelSettings(session, "label")
        if session.ui.is_gui:
            session.ui.triggers.add_handler('ready',
                lambda *args, ses=session: settings.register_settings_options(ses))

    @staticmethod
    def start_tool(session, tool_name):
        if tool_name == 'Scale Bar':
            from .scalebar_gui import Scalebar
            return Scalebar.get_singleton(session)

    @staticmethod
    def register_command(command_name, logger):
        # 'register_command' is lazily called when the command is referenced
        from . import label2d, label3d, arrows, scalebar
        label2d.register_label_command(logger)
        label3d.register_label_command(logger)
        arrows.register_arrow_command(logger)
        scalebar.register_scalebar_command(logger)

    @staticmethod
    def get_class(class_name):
        # 'get_class' is called by session code to get class saved in a session
        if class_name == 'ObjectLabels':
            from .label3d import ObjectLabels
            return ObjectLabels
        elif class_name == 'Labels':
            from .label2d import Labels
            return Labels
        elif class_name == 'LabelModel':
            from .label2d import LabelModel
            return LabelModel
        elif class_name == 'Arrows':
            from .arrows import Arrows
            return Arrows
        return None

bundle_api = _LabelBundle()

from .label2d import label_create, label_change, label_delete, Label, find_label
from .mouselabel import LabelMouseMode
from .movelabel import MoveLabelMouseMode
