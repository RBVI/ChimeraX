# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2024 Regents of the University of California.
# All rights reserved. This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use. For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===
from chimerax.core.commands import CmdDesc, register, BoolArg, EnumOf
from chimerax.core.models import REMOVE_MODELS

from chimerax.map import Volume
from chimerax.dicom.dicom_volumes import DICOMVolume

from chimerax.segmentations.view.ui import views, FourPanelView
from chimerax.segmentations.view.modes import ViewMode
from chimerax.segmentations.ui import find_segmentation_tool


def any_open_volumes(session) -> bool:
    return any(isinstance(v, Volume) for v in session.models)


def view_layout(
    session, layout: str = None, guidelines: bool = None, force=False
) -> None:
    st = find_segmentation_tool(session)
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


def _check_rapid_access(session, *_):
    if session.ui.is_gui:
        if not any_open_volumes(session):
            if session.ui.main_window.view_layout != "default":
                session.ui.main_window.restore_default_main_view()
            st = find_segmentation_tool()
            if st:
                st.delete()


def register_view_cmds(logger):
    register("ui view", view_layout_desc, view_layout, logger=logger)


def register_view_triggers(session):
    session.triggers.add_handler(
        REMOVE_MODELS, lambda *args, ses=session: _check_rapid_access(ses, *args)
    )
