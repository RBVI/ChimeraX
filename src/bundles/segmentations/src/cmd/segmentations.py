from typing import Optional, Union, Annotated

from chimerax.core.commands import CmdDesc, ModelIdArg, EnumOf, register, run
from chimerax.ui.cmd import ui_tool_show


from chimerax.segmentations.ui.segmentation_mouse_mode import (
    save_mouse_bindings,
    restore_mouse_bindings,
    save_hand_bindings,
    restore_hand_bindings,
)
from chimerax.segmentations.settings import get_settings

actions = [
    # "add",
    # "remove",
    "setMouseModes",
    "resetMouseModes",
    "setHandModes",
    "resetHandModes",
]


def segmentations(
    session,
    action,
):
    # model_specifier=None,
    # center: Optional[
    #    Union[
    #        # Spherical segmentations
    #        tuple[int, int, int],
    #        Annotated[list[int], 3],
    #        # Axial, Coronal, Sagittal slice segmentations
    #        tuple[int, int],
    #        Annotated[list[int], 2],
    #    ]
    # ] = None,
    # radius: Optional[int] = None,
    # minIntensity: Optional[int] = None,
    # maxIntensity: Optional[int] = None,
    # openTool: Optional[bool] = False,
    """Set or restore hand modes"""  # ; or add, delete, or modify segmentations."""
    settings = get_settings(session)
    # if session.ui.is_gui:
    #    from chimerax.segmentations.ui import find_segmentation_tool

    #    tool = find_segmentation_tool(session)
    #    if not tool and openTool:
    #        tool = get_segmentation_tool(session)
    # if action == "create":
    #    if not model_specifier:
    #        raise UserError("No model specified")
    #    if session.ui.is_gui:
    #        tool.addSegment()
    #    else:
    #        ...
    # if action == "delete":

    #    if not model_specifier:
    #        raise UserError("No model specified")
    #    if session.ui.is_gui:
    #        tool.addSegment()
    #    else:
    #        ...
    # if action == "add":
    #    if not model_specifier:
    #        raise UserError("No segmentation specified")
    # elif action == "remove":
    #    if not model_specifier:
    #        raise UserError("No segmentation specified")
    if action == "setMouseModes":
        save_mouse_bindings(session)
        run(session, "ui mousemode shift wheel 'resize segmentation cursor'")
        run(session, "ui mousemode right 'create segmentations'")
        run(session, "ui mousemode shift right 'erase segmentations'")
        run(session, "ui mousemode shift middle 'move segmentation cursor'")
    elif action == "resetMouseModes":
        restore_mouse_bindings(session)
    elif action == "setHandModes":
        save_hand_bindings(session, settings.vr_handedness)
        if settings.vr_handedness == "right":
            offhand = "left"
        else:
            offhand = "right"
        run(
            session,
            f"vr button b 'erase segmentations' hand { str(settings.vr_handedness).lower() }",
        )
        run(
            session,
            f"vr button a 'create segmentations' hand { str(settings.vr_handedness).lower() }",
        )
        run(session, f"vr button x 'toggle segmentation visibility' hand { offhand }")
        run(
            session,
            f"vr button thumbstick 'resize segmentation cursor' hand { str(settings.vr_handedness).lower() }",
        )
        run(
            session,
            f"vr button grip 'move segmentation cursor' hand { str(settings.vr_handedness).lower() }",
        )
    elif action == "resetHandModes":
        restore_hand_bindings(session)

segmentations_desc = CmdDesc(
    required=[("action", EnumOf(actions))],
    # optional=[("specifier", ModelIdArg)],
    synopsis="Set the view window to a grid of orthoplanes or back to the default",
)


def register_seg_cmds(logger):
    register(
        "segmentations",
        segmentations_desc,
        segmentations,
        logger=logger,
    )
