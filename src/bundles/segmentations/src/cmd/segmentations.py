from typing import Optional
from typing_extensions import Union
from chimerax.core.commands import (
    CmdDesc,
    ModelIdArg,
    IntArg,
    BoolArg,
    EnumOf,
    Or,
    TupleOf,
    register,
)
from chimerax.core.errors import UserError
from chimerax.ui.cmd import ui_tool_show
from chimerax.segmentations.ui.segmentations import SegmentationTool

actions = ["add", "remove", "create", "delete"]


# segmentations create #1
# segmentations delete #1
# semgentations add #1 axis axial center X Y radius Z
# semgentations add #1 center X Y Z radius Z
#
def segmentations(
    session,
    action,
    specifier=None,
    center: Optional[
        Union[
            # Spherical segmentations
            tuple[int, int, int],
            list[int, int, int],
            # Axial, Coronal, Sagittal slice segmentations
            tuple[int, int],
            list[int, int],
        ]
    ] = None,
    radius: Optional[int] = None,
    openTool: Optional[bool] = False,
):
    """Set the view window to a grid of orthoplanes or back to the default"""
    if session.ui.is_gui:
        tool = [t for t in session.tools if type(t) is SegmentationTool]
        if not tool and openTool:
            tool = open_segmentation_tool(session)
    if action == "create":
        if not specifier:
            raise UserError("No model specified")
        if session.ui.is_gui:
            tool.addSegment()
        else:
            ...
    if action == "delete":
        if not specifier:
            raise UserError("No model specified")
        if session.ui.is_gui:
            tool.addSegment()
        else:
            ...
    if action == "add":
        if not specifier:
            raise UserError("No segmentation specified")
    elif action == "remove":
        if not specifier:
            raise UserError("No segmentation specified")


def open_segmentation_tool(session):
    ui_tool_show(session, "segmentations")
    tlist = [t for t in session.tools if type(t) is SegmentationTool]
    tool = tlist[0]
    return tool


segmentations_desc = CmdDesc(
    required=[("action", EnumOf(actions)), ("specifier", ModelIdArg)],
    keyword=[
        ("center", Or(TupleOf(IntArg, 2), TupleOf(IntArg, 3))),
        ("radius", IntArg),
        ("openTool", BoolArg),
    ],
    synopsis=segmentations.__doc__.split("\n")[0],
)


def register_seg_cmds(logger):
    register(
        "segmentations",
        segmentations_desc,
        segmentations,
        logger=logger,
    )
