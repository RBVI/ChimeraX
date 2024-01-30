from chimerax.core.commands import CmdDesc, register, BoolArg, EnumOf
from chimerax.core.models import REMOVE_MODELS

from chimerax.map import Volume
from chimerax.dicom.dicom_volumes import DICOMVolume

from ..ui.view import views, FourPanelView
from ..ui.segmentations import SegmentationTool


def any_open_volumes(session) -> bool:
    return any(isinstance(v, Volume) for v in session.models)


def view_layout(
    session, layout: str = None, guidelines: bool = None, force=False
) -> None:
    st = None
    for tool in session.tools:
        if type(tool) == SegmentationTool:
            st = tool
            break
    if st:
        st.set_view_dropdown(layout)
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


view_layout_desc = CmdDesc(
    required=[("layout", EnumOf(["default", *views]))],
    keyword=[("guidelines", BoolArg)],
    optional=[("force", BoolArg)],
    synopsis="Set the view window to a grid of orthoplanes or back to the default",
)


def _check_rapid_access(*args):
    try:
        # This trigger fires many times, and on the last firing there is no model
        # we can pull the session out of, so we just have to catch the error here
        session = args[1][0].session
        if session.ui.is_gui:
            if not any_open_volumes(session):
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
    register("ui view", view_layout_desc, view_layout, logger=logger)
    logger.session.triggers.add_handler(REMOVE_MODELS, _check_rapid_access)
