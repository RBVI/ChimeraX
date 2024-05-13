def find_segmentation_tool(session):
    from chimerax.segmentations.ui.segmentations import SegmentationTool

    for tool in session.tools:
        if isinstance(tool, SegmentationTool):
            return tool
    return None
