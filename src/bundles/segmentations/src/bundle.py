from chimerax.core.toolshed import BundleAPI
from chimerax.segmentations.segmentation import Segmentation, open_grids_as_segmentation


class SegmentationsBundle(BundleAPI):
    api_version = 1

    @staticmethod
    def initialize(session, bundle_info):
        _ = bundle_info
        from chimerax.segmentations.segmentation_tracker import register_model_trigger_handlers

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
        register_model_trigger_handlers(session)

    @staticmethod
    def get_class(name):
        class_names = {
            "Segmentation": Segmentation,
            # "SegmentationJob": SegmentationJob
        }
        return class_names.get(name, None)

    @staticmethod
    def start_tool(*args):
        session, _, ti = args
        if ti.name == "Segmentations":
            from chimerax.segmentations.ui.segmentations import SegmentationTool
            return SegmentationTool(session)

        if ti.name == "Orthoplanes":
            from chimerax.segmentations.ui.orthoplane_tool import OrthoplaneTool
            return OrthoplaneTool(session)

    @staticmethod
    def register_command(*args):
        _, ci, logger = args
        if ci.name == "ui view":
            from chimerax.segmentations.view.cmd import register_view_cmds

            register_view_cmds(logger)
        elif ci.name == "segmentations":
            from chimerax.segmentations.cmd.segmentations import register_seg_cmds

            register_seg_cmds(logger)

    @staticmethod
    def run_provider(session, name, mgr, **_):
        if mgr == session.toolbar:
            from chimerax.segmentations.actions import run_toolbar_button

            return run_toolbar_button(session, name)
