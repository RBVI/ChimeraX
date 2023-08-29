# vim: set expandtab ts=4 sw=4:

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
from chimerax.mouse_modes import MouseMode
from .segmentations import SegmentationTool

class Segmentation3DMouseMode(MouseMode):
    name = 'vr segmentations'
    icon_file = 'segmentations.png'
    passively_follows_mouse = False
    # TODO: pasively_follows_hands = True

    def __init__(self, session):
        # We cannot check for the existence or lack thereof of the Segmentation Tool
        # here. When mouse modes are being registered, the session has not yet instantiated
        # whatever goes in its '.tools' attribute.
        MouseMode.__init__(self, session)
        self.segmentation_tool = None

    #@property
    #def settings(self):
    #    pass

    def enable(self):
        for tool in self.session.tools:
            if isinstance(tool, SegmentationTool):
                self.segmentation_tool = tool
                break

    def _find_segmentation_tool(self):
        for tool in self.session.tools:
            if isinstance(tool, SegmentationTool):
                return tool
        return None

    def mouse_down(self, event):
        MouseMode.mouse_down(self, event)
        #self.segmentation_tool = ...

    def wheel(self, event):
        d = event.wheel_value()
        if d > 0:
            self.segmentation_tool.segmentation_sphere.radius += 1
        elif d < 0:
            self.segmentation_tool.segmentation_sphere.radius -= 1

    def mouse_drag(self, event):
        if self.segmentation_tool is None:
            self.segmentation_tool = self._find_segmentation_tool()
        if self.segmentation_tool is None:
            return
        dx, dy = self.mouse_motion(event)
        #settings = self.settings
        ## Compute motion in scene coords of sphere center.
        c = self.segmentation_tool.segmentation_sphere.scene_position.origin()
        v = self.session.main_view
        s = v.pixel_size(c)
        if event.shift_down():
            shift = (0,0,s*dy)	# Move in z if shift key held.
        else:
            shift = (s*dx, -s*dy, 0)
        dxyz = v.camera.position.transform_vector(shift)
        self.segmentation_tool.move_sphere(dxyz)
        self.segmentation_tool.setSphereRegionToValue(
            self.segmentation_tool.segmentation_sphere.scene_position.origin()
            , self.segmentation_tool.segmentation_sphere.radius
            , 1
        )

    def mouse_up(self, event):
        MouseMode.mouse_up(self, event)

    def vr_motion(self, event):
        c = self.segmentation_tool.segmentation_sphere.scene_position.origin()
        delta_xyz = event.motion*c - c
        self.segmentation_tool.move_sphere(delta_xyz)