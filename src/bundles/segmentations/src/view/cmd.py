from chimerax.core.commands import CmdDesc, register, BoolArg, EnumOf
from chimerax.core.models import REMOVE_MODELS

from chimerax.map import Volume

import chimerax.segmentations.triggers
from chimerax.segmentations.triggers import Trigger

from chimerax.segmentations.view import views, FourPanelView
from chimerax.segmentations.ui import find_segmentation_tool
from chimerax.segmentations.settings import get_settings


def any_open_volumes(session) -> bool:
    return any(isinstance(v, Volume) for v in session.models)


def view_layout(
    session, layout: str = None, guidelines: bool = None, force=False
) -> None:
    # TODO Handle VR
    st = find_segmentation_tool(session)
    settings = get_settings(session)
    if layout == "default" and session.ui.main_window.view_layout != "default":
        session.ui.main_window.restore_default_main_view()
        chimerax.segmentations.triggers.activate_trigger(Trigger.ViewLayoutChanged, layout)
    elif layout in views and session.ui.main_window.view_layout != "orthoplanes":
        if not layout:
            session.ui.main_window.main_view = FourPanelView(session)
        else:
            session.ui.main_window.main_view = FourPanelView(session, layout)
        session.ui.main_window.view_layout = "orthoplanes"
        chimerax.segmentations.triggers.activate_trigger(Trigger.ViewLayoutChanged, layout)
        if st:
            session.ui.main_window.main_view.register_segmentation_tool(st)
        if guidelines:
            settings.display_guidelines = guidelines
            chimerax.segmentations.triggers.activate_trigger(
                Trigger.GuidelinesVisibilityChanged
            )
    elif layout in views and session.ui.main_window.view_layout == "orthoplanes":
        if layout:
            session.ui.main_window.main_view.convert_to_layout(layout)
            chimerax.segmentations.triggers.activate_trigger(
                Trigger.ViewLayoutChanged, layout
            )
        if guidelines is not None:
            settings.display_guidelines = guidelines
            chimerax.segmentations.triggers.activate_trigger(
                Trigger.GuidelinesVisibilityChanged
            )


view_layout_desc = CmdDesc(
    required=[("layout", EnumOf(["default", *views]))],
    keyword=[("guidelines", BoolArg)],
    optional=[("force", BoolArg)],
    synopsis="Set the view window to a grid of orthoplanes or back to the default",
)


def _check_rapid_access(session, *_):
    if session.ui.is_gui:
        if not any_open_volumes(session):
            if session.ui.main_window.view_layout != "default":
                session.ui.main_window.restore_default_main_view()
            st = find_segmentation_tool(session)
            if st:
                st.delete()


def register_view_cmds(logger):
    register("ui view", view_layout_desc, view_layout, logger=logger)


def register_view_triggers(session):
    session.triggers.add_handler(
        REMOVE_MODELS, lambda *args, ses=session: _check_rapid_access(ses, *args)
    )
