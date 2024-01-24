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
__version__ = "1.2"
from chimerax.core.toolshed import BundleAPI

from .segmentation import Segmentation


class _DICOMBundle(BundleAPI):
    api_version = 1

    @staticmethod
    def initialize(session, _):
        if session.ui.is_gui:
            from .ui.segmentation_mouse_mode import (
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

    @staticmethod
    def get_class(class_name):
        class_names = {
            "Segmentation": Segmentation,
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
            from .cmd.view import register_view_cmds

            register_view_cmds(logger)
            # elif ci.name == "segmentations":
            #    from .cmd.segmentations import register_seg_cmds

            #    register_seg_cmds(logger)


bundle_api = _DICOMBundle()
