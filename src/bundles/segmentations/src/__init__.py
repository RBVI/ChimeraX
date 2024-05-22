# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2024 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===
__version__ = "3.0.7"
from chimerax.core.toolshed import BundleAPI

from .segmentation import Segmentation, open_grids_as_segmentation


class _SegmentationsBundle(BundleAPI):
    api_version = 1

    @staticmethod
    def initialize(session, _):
        from .trigger_handlers import register_trigger_handlers

        if session.ui.is_gui:
            from chimerax.segmentations.view.cmd import register_view_triggers
            from chimerax.segmentations.ui.segmentation_mouse_mode import (
                CreateSegmentation3DMouseMode,
                EraseSegmentation3DMouseMode,
                Move3DSegmentationSphereMouseMode,
                Resize3DSegmentationSphereMouseMode,
                Toggle3DSegmentationVisibilityMouseMode,
            )

            for mode in [
                CreateSegmentation3DMouseMode,
                EraseSegmentation3DMouseMode,
                Move3DSegmentationSphereMouseMode,
                Resize3DSegmentationSphereMouseMode,
                Toggle3DSegmentationVisibilityMouseMode,
            ]:
                session.ui.mouse_modes.add_mode(mode(session))
            register_view_triggers(session)
        register_trigger_handlers(session)

    @staticmethod
    def get_class(class_name):
        class_names = {
            "Segmentation": Segmentation,
            # "SegmentationJob": SegmentationJob
        }
        return class_names.get(class_name, None)

    @staticmethod
    def start_tool(session, _, ti):
        if ti.name == "Segmentations":
            from .ui.segmentations import SegmentationTool

            return SegmentationTool(session)

    @staticmethod
    def register_command(_, ci, logger):
        if ci.name == "ui view":
            from chimerax.segmentations.view.cmd import register_view_cmds

            register_view_cmds(logger)
        elif ci.name == "segmentations":
            from .cmd.segmentations import register_seg_cmds

            register_seg_cmds(logger)

    @staticmethod
    def run_provider(session, name, mgr, **_):
        if mgr == session.toolbar:
            from .actions import run_toolbar_button

            return run_toolbar_button(session, name)


bundle_api = _SegmentationsBundle()
