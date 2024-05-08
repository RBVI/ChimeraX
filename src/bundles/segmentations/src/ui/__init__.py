from chimerax.segmentations.ui.segmentations import SegmentationTool


def find_segmentation_tool(session):
    for tool in session.tools:
        if isinstance(tool, SegmentationTool):
            return tool
    return None
