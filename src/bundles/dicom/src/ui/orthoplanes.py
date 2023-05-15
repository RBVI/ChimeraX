# vim: set expandtab shiftwidth=4 softtabstop=4:
# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California.
# All rights reserved. This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use. For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===
import math
from numpy import array, float32, uint8, int32
import numpy as np

from Qt.QtCore import Qt
from Qt.QtWidgets import (
    QVBoxLayout, QHBoxLayout
    , QWidget, QSlider
)

from Qt.QtGui import QWindow, QSurface
from chimerax.geometry import Place
from ..graphics import OrthoplaneView
from chimerax.graphics import MonoCamera, Drawing
from chimerax.graphics.opengl import GL
from chimerax.core.models import Surface
from chimerax.map import Volume, VolumeSurface, VolumeImage
from chimerax.ui.widgets import ModelMenu
from .label import Label
from enum import Enum

AXIAL_PLANE_MOVED = "axial plane moved"
CORONAL_PLANE_MOVED = "coronal plane moved"
SAGGITAL_PLANE_MOVED = "saggital plane moved"
# ORTHO_MODEL_CHANGED = "ortho model changed"

orthoplane_triggers = [
    AXIAL_PLANE_MOVED, CORONAL_PLANE_MOVED, SAGGITAL_PLANE_MOVED
]

class Axis(Enum):
    AXIAL = 2
    CORONAL = 1
    SAGGITAL = 0

    def __str__(self):
        return self.name.lower()

class OrthoplaneGraphicsWindow(QWindow):

    EyeSize = 4     # half size really

    ON_NOTHING = 0
    ON_EYE = 1
    ON_NEAR = 2
    ON_FAR = 3

    def __init__(self, parent, session, panel, axis = Axis.AXIAL):
        QWindow.__init__(self)
        from Qt.QtWidgets import QWidget
        self.moving = self.ON_NOTHING
        self.session = session
        self.axis = axis
        self.overlays = []
        self.slider_moved = False
        self.last_mouse_position = None
        self.widget = QWidget.createWindowContainer(self, parent)
        self.setSurfaceType(QSurface.SurfaceType.OpenGLSurface)
        self.view = OrthoplaneView(Drawing("placeholder"), window_size = (0, 0), axis = self.axis.value)
        self.view.initialize_rendering(session.main_view.render.opengl_context)
        # TODO: from chimerax.graphics.camera import OrthographicCamera
        self.view.camera = MonoCamera()
        if self.axis == Axis.AXIAL:
            self.axes = [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]
        elif self.axis == Axis.CORONAL:
            self.axes = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
        else:
            self.axes = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
        camera = self.view.camera
        camera.position = Place(origin = (0,0,0), axes = self.axes)
        self.view.background_color = (255, 255, 255, 255)
        self.panel = panel
        self.main_view = session.main_view
        self.camera_offsets = [0, 0, 0]
        self.label = Label(self.session, self.view, str(axis), str(axis), size=16, xpos=0, ypos=0)
        self.segmentation_slices = {}

        max_x, max_y, max_z = 2, 2, 2

        self.segmentation_overlay = SegmentationOverlay("seg_overlay", radius=10)
        self.view.add_overlay(self.segmentation_overlay)
        self.slider = QSlider(Qt.Orientation.Horizontal, parent)
        self.old_pos = 0
        self.pos = 0
        self.x = 0
        self.y = 0
        if axis == Axis.AXIAL:
            self.slider.setMaximum(max_z)
            self.slider.setValue(1)
        if axis == Axis.CORONAL:
            self.slider.setRange(-max_y, 0)
            self.slider.setValue(-1)
        if axis == Axis.SAGGITAL:
            self.slider.setMaximum(max_x)
            self.slider.setValue(1)

        self.slider.sliderMoved.connect(self._on_slider_moved)

        self.handler = session.triggers.add_handler('frame drawn', self._redraw)
        from Qt.QtCore import QSize
        self.widget.setMinimumSize(QSize(20, 20))

    def _on_slider_moved(self):
        if self.axis == Axis.CORONAL:
            self.old_pos = -self.pos
            self.pos = -self.slider.sliderPosition()
            diff = self.old_pos + self.pos
        else:
            self.old_pos = self.pos
            self.pos = self.slider.sliderPosition()
            diff = self.pos - self.old_pos
        self.slider_moved = True
        if self.axis == Axis.AXIAL:
            # TODO: Convert this object to use the Volume and not the VolumeImage?
            # TODO: Add an API to Volume and Grid to get underlying data?
            # TODO: DICOM, NRRD, and NIfTI need mutually compatible methods
            if self.view.drawing.parent.data.dicom_data.inferior_to_superior:
                diff = -diff
            self.camera_offsets[2] += diff * self.view.drawing.parent.data.step[2]
        if self.axis == Axis.CORONAL:
            self.camera_offsets[1] += diff * self.view.drawing.parent.data.step[1]
        if self.axis == Axis.SAGGITAL:
            self.camera_offsets[0] -= diff * self.view.drawing.parent.data.step[0]
        self.view.camera.redraw_needed = True

    def close(self):
        self.session.triggers.remove_handler(self.handler)
        # TODO: why does this call make it crash?
        # self.setParent(None)
        self.label.delete()
        del self.label
        self.view.delete()
        QWindow.destroy(self)

    def add_overlay(self, overlay):
        self.overlays.append(overlay)

    def delete_overlay(self, overlay):
        self.overlays.remove(overlay)
        overlay.delete()

    def _redraw(self, *_):
        self.render()

    def exposeEvent(self, event):
        if self.isExposed() and not self.session.update_loop.blocked():
            self.render()

    def resizeEvent(self, event):  # noqa
        size = event.size()
        width = size.width()
        height = size.height()
        self.set_viewport(width, height)

    def set_viewport(self, width, height):
        # Don't need make_current, since OpenGL isn't used
        # until rendering
        self.view.resize(width, height)

    def render(self):
        ww, wh = self.main_view.window_size
        width, height = self.view.window_size
        if ww <= 0 or wh <= 0 or width <= 0 or height <= 0:
            return
        # temporary workaround for #2162
        if self.view is None or self.view.render is None:
            return
        mvwin = self.view.render.use_shared_context(self)
        try:
            # TODO: Set the clip planes for the camera to be very far away. Some DICOMs are huge
            # and require large zoom-outs to get them into view
            if any(x.devicePixelRatio() >= 2.0 for x in self.session.ui.screens()):
                GL.glPointSize(3)
            old_disp_val = self.view.drawing.display
            if not old_disp_val:
                self.view.drawing.display = True
            # TODO: If the user selects 'surface' then 'orthoplanes' in the volume viewer we should
            # override the default plane locations somehow
            if self.slider_moved:
                orthoplane_positions = list(self.view.drawing._rendering_options.orthoplane_positions)
                orthoplane_positions[self.axis.value] = self.pos
                self.view.drawing.parent.set_parameters(orthoplane_positions=orthoplane_positions)
                self.view.drawing.parent.update_drawings()
                self.slider_moved = False
            bounds = self.view.drawing.bounds()
            # We use these offsets to align the camera to the actual center of the orthoplanes
            model_x_offset, model_y_offset, model_z_offset = bounds.center()
            x_size, y_size, z_size = bounds.size()
            x_max, y_max, z_max = bounds.xyz_max
            cameraView = 2 * math.tan(0.5 * math.radians(self.view.camera.field_of_view))
            # TODO: Relate this to the size of the data on each axis these values are only good for the RIDER CT
            if self.axis == Axis.AXIAL:
                cameraDistance = 1.9
            if self.axis == Axis.CORONAL:
                cameraDistance = 1.5
            if self.axis == Axis.SAGGITAL:
                cameraDistance = 1.5
            x_apparent, y_apparent, z_apparent = [
                x_size * cameraDistance / cameraView
                , y_size * cameraDistance / cameraView
                , z_size * cameraDistance / cameraView
            ]
            # Set initial camera view
            camera_x_offset, camera_y_offset, camera_z_offset = self.camera_offsets
            self.origin = self.view.drawing.position.origin()
            if self.axis == Axis.AXIAL:
                self.origin += [
                     model_x_offset + camera_x_offset
                    , model_y_offset + camera_y_offset
                    , model_z_offset + (z_apparent - camera_z_offset)
                ]
            elif self.axis == Axis.CORONAL:
                self.origin += [
                    model_x_offset + camera_x_offset
                    , model_y_offset - (y_apparent - camera_y_offset)
                    , model_z_offset + camera_z_offset
                ]
            else:
                self.origin += [
                     model_x_offset + (x_apparent - camera_x_offset)
                    , model_y_offset + camera_y_offset
                    , model_z_offset + camera_z_offset
                ]
            camera = self.view.camera
            camera.position = Place(axes=self.axes, origin=self.origin)
            self.segmentation_overlay.update()

            self.view.prepare_scene_for_drawing()
            self.view._draw_scene(self.view.camera, [self.view.drawing])
            self.view.finalize_draw()
            self.view.drawing.display = old_disp_val
            GL.glPointSize(1)
        except Exception as e:
            # This line is here so you can set a breakpoint on it and figure out what's going wrong
            # because ChimeraX's interface will not tell you.
            pass
        finally:
            # Target opengl context back to main graphics window.
            self.main_view.render.use_shared_context(mvwin)
        self.view.render.done_current()

    def mousePressEvent(self, event):  # noqa
        b = event.button() | event.buttons()
        if b & Qt.MouseButton.RightButton:
            # p = event.position() if hasattr(event, 'position') else event.pos()  # PyQt6 / PyQt5
            from Qt.QtGui import QContextMenuEvent
            e = QContextMenuEvent(QContextMenuEvent.Mouse, event.pos())
            self.widget.parent().parent().contextMenuEvent(e)
            return
        if b & Qt.MouseButton.MiddleButton:
            return
        if b & Qt.MouseButton.LeftButton:
            # Whatever is needed to start segmenting
            return

    def mouseReleaseEvent(self, event): # noqa
        b = event.button() | event.buttons()
        if b & Qt.MouseButton.LeftButton:
            self.segmentation_overlay.center = (2 * event.position().x(), 2*(self.view.window_size[1] - event.position().y()), 0)
            self.segmentation_overlay.update()
            self.view.camera.redraw_needed = True
        self.last_mouse_position = None

    def wheelEvent(self, event):
        modifier = event.modifiers()
        if modifier == Qt.KeyboardModifier.ShiftModifier:
            self.segmentation_overlay.radius += 1 * np.sign(event.angleDelta().y())
        elif modifier == Qt.KeyboardModifier.NoModifier:
            self.camera_offsets[self.axis.value] += 15 * np.sign(event.angleDelta().y())
        self.view.camera.redraw_needed = True

    def mouseMoveEvent(self, event):  # noqa
        b = event.button() | event.buttons()
        # Level or segment
        if b == Qt.MouseButton.NoButton or b == Qt.MouseButton.LeftButton:
            self.segmentation_overlay.center = (2 * event.position().x(), 2* (self.view.window_size[1] - event.position().y()), 0)
            self.segmentation_overlay.update()
            self.view.camera.redraw_needed = True
            return
        # Zoom / Dolly
        if b & Qt.MouseButton.RightButton:
            x, y = event.position().x(), event.position().y()
            if not self.last_mouse_position:
                dy = 0
            else:
                dy = y - self.last_mouse_position[1]
            psize = self.view.pixel_size()
            self.last_mouse_position = [x, y]
            self.camera_offsets[self.axis.value] += (-dy * psize) * 3
            self.view.camera.redraw_needed = True
        # Truck & Pedestal
        if b & Qt.MouseButton.MiddleButton:
            x, y = event.position().x(), event.position().y()
            psize = self.view.pixel_size()
            if not self.last_mouse_position:
                dx = 0
                dy = 0
            else:
                dx = (x - self.last_mouse_position[0]) * psize
                dy = (y - self.last_mouse_position[1]) * psize
            self.last_mouse_position = [x, y]
            x, y, z = self.camera_offsets
            if self.axis == Axis.AXIAL:
                self.camera_offsets = [x + dx, y - dy, z]
            if self.axis == Axis.CORONAL:
                self.camera_offsets = [x - dx, y, z + dy]
            if self.axis == Axis.SAGGITAL:
                self.camera_offsets = [x, y - dx, z + dy]

    # def touchEvent(self, event):
    #     pass

    def keyPressEvent(self, event):  # noqa
        return self.session.ui.forward_keystroke(event)

    def _surface_chosen(self, *args):
        new_drawing = None
        for d in self.panel.model_menu.value._child_drawings:
            if type(d) is VolumeImage:
                new_drawing = d
        if new_drawing is not None:
            self.view.drawing = new_drawing
            self.label.text = d.parent.name
            self.label.update_drawing()
            max_x, max_y, max_z = max_slider_vals = self.view.drawing._region[1]
            orthoplane_positions = self.view.drawing._rendering_options.orthoplane_positions
            if self.axis == Axis.AXIAL:
                self.slider.setMaximum(max_z)
                self.slider.setValue(orthoplane_positions[2])
                self.pos = orthoplane_positions[2]
            if self.axis == Axis.CORONAL:
                self.slider.setRange(-max_y, 0)
                self.slider.setValue(-orthoplane_positions[1])
                self.pos = orthoplane_positions[1]
            if self.axis == Axis.SAGGITAL:
                self.slider.setMaximum(max_x)
                self.slider.setValue(orthoplane_positions[0])
                self.pos = orthoplane_positions[0]
            self.view.camera.redraw_needed = True


class SegmentationOverlay(Drawing):
    def __init__(self, name, radius):
        super().__init__(name)
        self.display_style = Drawing.Dot
        self.use_lighting = False
        self._radius = radius
        self._center = [0, 0, 0]

    def draw(self, renderer, draw_pass):
        r = renderer
        ww, wh = r.render_size()
        from chimerax.graphics.camera import ortho
        projection = ortho(0, ww, 0, wh, -1, 1)
        r.set_projection_matrix(projection)
        Drawing.draw(self, renderer, draw_pass)
        r.set_projection_matrix(
            ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0),
             (0, 0, 0, 1))
            )

    @property
    def center(self):
        return self._center

    @center.setter
    def center(self, center):
        self._center = center

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, radius):
        self._radius = radius

    def update(self):
        vc, v, _, t = self._geometry()
        self.set_geometry(v, _, t)
        self.vertex_colors = vc

    def _geometry(self):
        # Bresenham's Algorithm
        def mirror_points_8(x, y):
            return [(x, y), (y, x), (-x, y), (-y, x), (x, -y), (y, -x), (-x, -y), (-y, -x)]
        x = 0
        y = self.radius
        d = 1 - y
        v = []
        v.extend(mirror_points_8(x, y))
        while y > x:
            if d < 0:
                d += 2*x + 3
            else:
                d += 2*(x - y) + 5
                y -= 1
            x += 1
            v.extend(mirror_points_8(x, y))
        fv = [[self.center[0] + vt[0], self.center[1] + vt[1], 0] for vt in v]
        # We don't use this but we must pass t along so compute it anyway
        t = []
        for i in range(0, len(v)):
            t.append([i, i + 1])
        t[0][1] = 0
        t[-1][1] = 0
        fv = array(fv, dtype=float32)
        t = array(t, dtype=int32)
        vc = array([[255, 0, 0, 255]] * len(v), dtype=uint8)
        return vc, fv, None, t

class PlaneViewer(QWidget):

    def __init__(self, session, axis, parent):
        self.session = session
        super().__init__(parent)
        # UI content code
        if axis == Axis.AXIAL:
            name = "axial orthoplane"
        elif axis == Axis.CORONAL:
            name = "coronal orthoplane"
        elif axis == Axis.SAGGITAL:
            name = "saggital orthoplane"
        else:
            name = "BADAXIS"
        self.name = name
        self.axis = axis
        self.opengl_canvas = OrthoplaneGraphicsWindow(parent, session, self, axis = self.axis)

        # From surface/areagui.py
        def _not_volume_surface(m):
            return not isinstance(m, VolumeSurface)

        self.model_menu = ModelMenu(
            self.session, parent, label = 'Model',
            model_types = [Volume, Surface],
            model_filter = _not_volume_surface,
            model_chosen_cb = self.opengl_canvas._surface_chosen
        )

        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(0)
        button_layout.addStretch(1)
        button_layout.addWidget(self.model_menu.frame)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addLayout(button_layout)
        self.container = QWidget(parent)
        container_layout = QVBoxLayout()
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)
        container_layout.addWidget(self.opengl_canvas.widget, stretch = 1)
        container_layout.addWidget(self.opengl_canvas.slider)
        self.container.setLayout(container_layout)

        layout.addWidget(self.container, stretch=1)
        self.setLayout(layout)


    def delete(self):
        self.opengl_canvas.close()
