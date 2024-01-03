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
from chimerax.map import add_map_format
from chimerax.core.tools import get_singleton
from .dicom import (
    DICOMMapFormat, DicomOpener, fetchers,
    Patient, Study
)
from .dicom.dicom_volumes import DICOMVolume

class _DICOMBundle(BundleAPI):
    api_version = 1

    @staticmethod
    def initialize(session, bundle_info):
        """Register file formats, commands, and database fetch."""
        add_map_format(session, DICOMMapFormat())
        if session.ui.is_gui:
            from .ui.segmentation_mouse_mode import (
                CreateSegmentation3DMouseMode, EraseSegmentation3DMouseMode, Move3DSegmentationSphereMouseMode
                , Resize3DSegmentationSphereMouseMode, Toggle3DSegmentationVisibilityMouseMode
            )
            for mode in [
                CreateSegmentation3DMouseMode, EraseSegmentation3DMouseMode, Move3DSegmentationSphereMouseMode
                , Resize3DSegmentationSphereMouseMode, Toggle3DSegmentationVisibilityMouseMode
            ]:
                session.ui.mouse_modes.add_mode(mode(session))

    @staticmethod
    def get_class(class_name):
        class_names = {
            'Patient': Patient
            , 'Study': Study
            , 'DICOMVolume': DICOMVolume
        }
        return class_names.get(class_name, None)

    @staticmethod
    def start_tool(session, bi, ti):
        if ti.name == "DICOM Browser":
            from .dicom.ui import DICOMBrowserTool
            return get_singleton(session, DICOMBrowserTool, "DICOM Browser")
        elif ti.name == "Segmentations":
            from .ui.segmentations import SegmentationTool
            return SegmentationTool(session)
        else:
            from .dicom.ui import DICOMDatabases
            return DICOMDatabases(session)

    @staticmethod
    def register_command(bi, ci, logger):
        if ci.name == "dicom view":
            from .cmd.view import register_view_cmds
            register_view_cmds(logger)
        elif ci.name == "dicom segmentations":
            from .cmd.segmentations import register_seg_cmds
            register_seg_cmds(logger)
        #elif ci.name == "monailabel":
        #    from .monailabel import register_cmds
        #    register_cmds(logger)

    @staticmethod
    def run_provider(session, name, mgr, **kw):
        # return runners['name']
        if name == "DICOM medical imaging":
            return DicomOpener()
        else:
            return fetchers[name]()


bundle_api = _DICOMBundle()
