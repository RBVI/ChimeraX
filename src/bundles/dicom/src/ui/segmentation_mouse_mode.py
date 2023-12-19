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
import os

from enum import Enum

from chimerax.core.settings import Settings
from chimerax.mouse_modes import MouseMode

from .segmentations import SegmentationTool


class CreateSegmentation3DMouseMode(MouseMode):
    """Use the segmentation sphere to mark off regions of data in 3D."""
    name = 'create segmentations'
    icon_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'icons', 'create_segmentation.png')
    def __init__(self, session):
        # We cannot check for the existence or lack thereof of the Segmentation Tool
        # here. When mouse modes are being registered, the session has not yet instantiated
        # whatever goes in its '.tools' attribute.
        MouseMode.__init__(self, session)
        self.segmentation_tool = None

    def enable(self):
        self.segmentation_tool = self._find_segmentation_tool()

    def _find_segmentation_tool(self):
        for tool in self.session.tools:
            if isinstance(tool, SegmentationTool):
                return tool
        return None

    def mouse_down(self, event):
        MouseMode.mouse_down(self, event)
        if self.segmentation_tool is None:
            self.segmentation_tool = self._find_segmentation_tool()
        if self.segmentation_tool is None:
            return
        self.segmentation_tool.set_segmentation_step(2)
        self.segmentation_tool.setSphereRegionToValue(
            self.segmentation_tool.segmentation_sphere.scene_position.origin()
            , self.segmentation_tool.segmentation_sphere.radius
            , 1
        )

    def wheel(self, event):
        if self.segmentation_tool is None:
            self.segmentation_tool = self._find_segmentation_tool()
        if self.segmentation_tool is None:
            return
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
        shift = (s*dx, -s*dy, 0)
        dxyz = v.camera.position.transform_vector(shift)
        if event.shift_down():
            shift = (s*dx, -s*dy, 0)
        else:
            shift = (s*dx, -s*dy, 0)
        self.segmentation_tool.move_sphere(dxyz)
        self.segmentation_tool.setSphereRegionToValue(
            self.segmentation_tool.segmentation_sphere.scene_position.origin()
            , self.segmentation_tool.segmentation_sphere.radius
            , 1
        )

    def vr_press(self, event):
        if self.segmentation_tool is None:
            self.segmentation_tool = self._find_segmentation_tool()
        if self.segmentation_tool is None:
            return
        self.segmentation_tool.set_segmentation_step(2)

    def vr_release(self, event):
        MouseMode.mouse_up(self, event)
        if self.segmentation_tool is None:
            self.segmentation_tool = self._find_segmentation_tool()
        if self.segmentation_tool is None:
            return
        self.segmentation_tool.set_segmentation_step(1)

    def mouse_up(self, event):
        MouseMode.mouse_up(self, event)
        if self.segmentation_tool is None:
            self.segmentation_tool = self._find_segmentation_tool()
        if self.segmentation_tool is None:
            return
        self.segmentation_tool.set_segmentation_step(1)

    def vr_motion(self, event):
        if self.segmentation_tool is None:
            self.segmentation_tool = self._find_segmentation_tool()
        if self.segmentation_tool is None:
            return
        c = self.segmentation_tool.segmentation_sphere.scene_position.origin()
        delta_xyz = event.motion*c - c
        self.segmentation_tool.move_sphere(delta_xyz)
        self.segmentation_tool.setSphereRegionToValue(
            self.segmentation_tool.segmentation_sphere.scene_position.origin()
            , self.segmentation_tool.segmentation_sphere.radius
            , 1
        )

class EraseSegmentation3DMouseMode(MouseMode):
    """Use the segmentation sphere to erase regions of data in 3D."""
    name = 'erase segmentations'
    icon_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'icons', 'create_segmentation.png')
    def __init__(self, session):
        # We cannot check for the existence or lack thereof of the Segmentation Tool
        # here. When mouse modes are being registered, the session has not yet instantiated
        # whatever goes in its '.tools' attribute.
        MouseMode.__init__(self, session)
        self.segmentation_tool = None

    def enable(self):
        self.segmentation_tool = self._find_segmentation_tool()

    def _find_segmentation_tool(self):
        for tool in self.session.tools:
            if isinstance(tool, SegmentationTool):
                return tool
        return None

    def mouse_down(self, event):
        MouseMode.mouse_down(self, event)
        if self.segmentation_tool is None:
            self.segmentation_tool = self._find_segmentation_tool()
        if self.segmentation_tool is None:
            return
        self.segmentation_tool.set_segmentation_step(2)
        self.segmentation_tool.setSphereRegionToValue(
            self.segmentation_tool.segmentation_sphere.scene_position.origin()
            , self.segmentation_tool.segmentation_sphere.radius
            , 0
        )

    def wheel(self, event):
        if self.segmentation_tool is None:
            self.segmentation_tool = self._find_segmentation_tool()
        if self.segmentation_tool is None:
            return
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
        shift = (s*dx, -s*dy, 0)
        dxyz = v.camera.position.transform_vector(shift)
        if event.shift_down():
            shift = (s*dx, -s*dy, 0)
        else:
            shift = (s*dx, -s*dy, 0)
        self.segmentation_tool.move_sphere(dxyz)
        self.segmentation_tool.setSphereRegionToValue(
            self.segmentation_tool.segmentation_sphere.scene_position.origin()
            , self.segmentation_tool.segmentation_sphere.radius
            , 0
        )

    def mouse_up(self, event):
        MouseMode.mouse_up(self, event)
        if self.segmentation_tool is None:
            self.segmentation_tool = self._find_segmentation_tool()
        if self.segmentation_tool is None:
            return
        self.segmentation_tool.set_segmentation_step(1)

    def vr_press(self, event):
        if self.segmentation_tool is None:
            self.segmentation_tool = self._find_segmentation_tool()
        if self.segmentation_tool is None:
            return
        self.segmentation_tool.set_segmentation_step(2)

    def vr_release(self, event):
        MouseMode.mouse_up(self, event)
        # Any positive Y reading indicates pushing up, getting bigger
        # Any negative Y reading indicates pushing down, getting smaller
        if self.segmentation_tool is None:
            self.segmentation_tool = self._find_segmentation_tool()
        if self.segmentation_tool is None:
            return
        self.segmentation_tool.set_segmentation_step(1)

    def vr_motion(self, event):
        # Any positive Y reading indicates pushing up, getting bigger
        # Any negative Y reading indicates pushing down, getting smaller
        if self.segmentation_tool is None:
            self.segmentation_tool = self._find_segmentation_tool()
        if self.segmentation_tool is None:
            return
        c = self.segmentation_tool.segmentation_sphere.scene_position.origin()
        delta_xyz = event.motion*c - c
        self.segmentation_tool.move_sphere(delta_xyz)
        self.segmentation_tool.setSphereRegionToValue(
            self.segmentation_tool.segmentation_sphere.scene_position.origin()
            , self.segmentation_tool.segmentation_sphere.radius
            , 0
        )


class Move3DSegmentationSphereMouseMode(MouseMode):
    name = 'move segmentation cursor'
    icon_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'icons', 'move_cursor.png')
    def __init__(self, session):
        MouseMode.__init__(self, session)
        self.segmentation_tool = None

    def enable(self):
        self.segmentation_tool = self._find_segmentation_tool()

    def _find_segmentation_tool(self):
        for tool in self.session.tools:
            if isinstance(tool, SegmentationTool):
                return tool
        return None

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
        shift = (s*dx, -s*dy, 0)
        dxyz = v.camera.position.transform_vector(shift)
        shift = (s*dx, -s*dy, 0)
        self.segmentation_tool.move_sphere(dxyz)

    def vr_motion(self, event):
        # Any positive Y reading indicates pushing up, getting bigger
        # Any negative Y reading indicates pushing down, getting smaller
        if self.segmentation_tool is None:
            self.segmentation_tool = self._find_segmentation_tool()
        if self.segmentation_tool is None:
            return
        c = self.segmentation_tool.segmentation_sphere.scene_position.origin()
        delta_xyz = event.motion*c - c
        self.segmentation_tool.move_sphere(delta_xyz)

class Toggle3DSegmentationVisibilityMouseMode(MouseMode):
    name = 'toggle segmentation visibility'
    #icon_path =

    def __init__(self, session):
        MouseMode.__init__(self, session)
        self.segmentaiton_tool = None

    def enable(self):
        self.segmentation_tool = self._find_segmentation_tool()

    def _find_segmentation_tool(self):
        for tool in self.session.tools:
            if isinstance(tool, SegmentationTool):
                return tool
        return None

    def vr_press(self, event):
        if self.segmentation_tool is None:
            self.segmentation_tool = self._find_segmentation_tool()
        if self.segmentation_tool is None:
            return
        self.segmentation_tool.hide_active_segmentation()


    def vr_release(self, event):
        if self.segmentation_tool is None:
            self.segmentation_tool = self._find_segmentation_tool()
        if self.segmentation_tool is None:
            return
        self.segmentation_tool.show_active_segmentation()



class Resize3DSegmentationSphereMouseMode(MouseMode):
    name = 'resize segmentation cursor'
    icon_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'icons', 'resize_cursor.png')
    def __init__(self, session):
        MouseMode.__init__(self, session)
        self.segmentation_tool = None

    def enable(self):
        self.segmentation_tool = self._find_segmentation_tool()


    def _find_segmentation_tool(self):
        for tool in self.session.tools:
            if isinstance(tool, SegmentationTool):
                return tool
        return None

    def wheel(self, event):
        if self.segmentation_tool is None:
            self.segmentation_tool = self._find_segmentation_tool()
        if self.segmentation_tool is None:
            return
        d = event.wheel_value()
        if d > 0:
            self.segmentation_tool.segmentation_sphere.radius += 0.25
        elif d < 0:
            self.segmentation_tool.segmentation_sphere.radius -= 0.25

    def vr_motion(self, event):
        if self.segmentation_tool is None:
            self.segmentation_tool = self._find_segmentation_tool()
        if self.segmentation_tool is None:
            return
        c = self.segmentation_tool.segmentation_sphere.scene_position.origin()
        delta_xyz = event.motion*c - c
        self.segmentation_tool.move_sphere(delta_xyz)

    def vr_thumbstick(self, event):
        # Any positive Y reading indicates pushing up, getting bigger
        # Any negative Y reading indicates pushing down, getting smaller
        if self.segmentation_tool is None:
            self.segmentation_tool = self._find_segmentation_tool()
        if self.segmentation_tool is None:
            return
        d = event.y
        if d > 0:
            self.segmentation_tool.segmentation_sphere.radius += 0.25
        elif d < 0:
            self.segmentation_tool.segmentation_sphere.radius -= 0.25