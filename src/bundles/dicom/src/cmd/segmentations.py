from chimerax.core.commands import CmdDesc, ModelIdArg, EnumOf, register
from ..ui.segmentations import SegmentationTool
from chimerax.ui.cmd import ui_tool_show

actions = ["add", "remove"]

def dicom_segmentations(session, action, specifier = None):
    tlist = [t for t in session.tools if type(t) is SegmentationTool]
    if not tlist:
        ui_tool_show(session, "segmentations")
        tlist = [t for t in session.tools if type(t) is SegmentationTool]
    tool = tlist[0]
    if action == "add":
        tool.addSegment()
    #elif action == "remove":
    #    if not specifier:
    #        ...


dicom_segmentations_desc = CmdDesc(
    required = [("action", EnumOf(actions))],
    optional=[
        ("specifier", ModelIdArg)
    ],
    synopsis = "Set the view window to a grid of orthoplanes or back to the default"
)

def register_seg_cmds(logger):
    register("dicom segmentations", dicom_segmentations_desc, dicom_segmentations, logger=logger)
