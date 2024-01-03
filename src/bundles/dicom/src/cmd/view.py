from chimerax.core.commands import StringArg, CmdDesc, register, BoolArg, EnumOf
from chimerax.core.models import REMOVE_MODELS

from chimerax.map import Volume
from chimerax.nifti import NiftiGrid
from chimerax.nrrd import NRRDGrid

from ..dicom import DicomGrid
from ..dicom.dicom_volumes import DICOMVolume
from ..ui.view import views, FourPanelView
from ..ui.segmentations import SegmentationTool

medical_types = [DicomGrid, NiftiGrid, NRRDGrid]

def dicom_view(session, layout: str = None, guidelines: bool = None, force = False) -> None:
    # TODO: Enable for NIfTI and NRRD as well
    open_volumes = [v for v in session.models if type(v) is Volume or type(v) is DICOMVolume]
    medical_volumes = [m for m in open_volumes if type(m.data) in medical_types]
    st = None
    for tool in session.tools:
        if type(tool) == SegmentationTool:
            st = tool
            break
    if st:
        st.set_view_dropdown(layout)
    if not force and not medical_volumes:
        session.logger.error("No medical images open")
        return
    if layout == "default" and session.ui.main_window.view_layout != "default":
        session.ui.main_window.restore_default_main_view()
    elif layout in views and session.ui.main_window.view_layout != "orthoplanes":
        if not layout:
            session.ui.main_window.main_view = FourPanelView(session)
        else:
            session.ui.main_window.main_view = FourPanelView(session, layout)
        session.ui.main_window.view_layout = "orthoplanes"
        if st:
            session.ui.main_window.main_view.register_segmentation_tool(st)
        if guidelines:
            session.ui.main_window.main_view.set_guideline_visibility(guidelines)
    elif layout in views and session.ui.main_window.view_layout == "orthoplanes":
        if layout:
            session.ui.main_window.main_view.convert_to_layout(layout)
        if guidelines is not None:
            session.ui.main_window.main_view.set_guideline_visibility(guidelines)


dicom_view_desc = CmdDesc(
    required = [("layout", EnumOf(["default", *views]))],
    keyword = [
        ("guidelines", BoolArg)
    ]
    , optional = [("force", BoolArg)],
    synopsis = "Set the view window to a grid of orthoplanes or back to the default"
)

def _check_rapid_access(*args):
    try:
        # This trigger fires many times, and on the last firing there is no model
        # we can pull the session out of, so we just have to catch the error here
        session = args[1][0].session
        any_open_models = any(type(v) == DICOMVolume or type(v) == Volume for v in session.models)
        if session.ui.is_gui:
            if not any_open_models:
                if session.ui.main_window.view_layout != "default":
                   session.ui.main_window.restore_default_main_view()
                st = None
                for tool in session.tools:
                    if type(tool) == SegmentationTool:
                        st = tool
                        break
                if st:
                    st.delete()
    except IndexError:
        pass

def register_view_cmds(logger):
    register("dicom view", dicom_view_desc, dicom_view, logger=logger)
    logger.session.triggers.add_handler(REMOVE_MODELS, _check_rapid_access)
