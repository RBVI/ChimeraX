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

from Qt.QtCore import Qt
from Qt.QtWidgets import (
    QVBoxLayout, QHBoxLayout
    , QWidget, QSlider
)

from Qt.QtGui import QWindow, QSurface
from chimerax.geometry import Place
from ..graphics import OrthoplaneView
from chimerax.graphics import MonoCamera, Drawing
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
        self.camera_offset = 0
        self.label = Label(self.session, self.view, str(axis), str(axis), size=16, xpos=0, ypos=0)

        max_x, max_y, max_z = 2, 2, 2

        class _PixelLocations():
            pass

        self.locations = loc = _PixelLocations()
        loc.eye = 0, 0, 0   # x, y coordinates of eye
        loc.near = 0        # X coordinate of near plane
        loc.far = 0         # Y coordinate of near plane
        loc.bottom = 0      # bottom of clipping planes
        loc.top = 0         # top of clipping planes
        loc.far_bottom = 0  # right clip intersect far
        loc.far_top = 0     # left clip intersect far


        self.segmentation_overlay = SegmentationOverlay("seg_overlay")
        self.view.add_overlay(self.segmentation_overlay)
        self.segmentation_radius = 10
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
        else:
            self.old_pos = self.pos
            self.pos = self.slider.sliderPosition()
        self.slider_moved = True
        #diff = self.pos - self.old_pos
        #if self.axis == Axis.AXIAL:
        #    self.camera_offset -= diff * self.view.drawing.parent.data.step[2]
        #if self.axis == Axis.CORONAL:
        #    self.camera_offset += diff * self.view.drawing.parent.data.step[1]
        #if self.axis == Axis.SAGGITAL:
        #    self.camera_offset -= diff * self.view.drawing.parent.data.step[0]
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
            old_disp_val = self.view.drawing.display
            if not old_disp_val:
                self.view.drawing.display = True
            if self.slider_moved:
                new_orthoplane_positions = old_orthoplane_positions = self.view.drawing._rendering_options.orthoplane_positions
                if self.axis == Axis.AXIAL:
                    new_orthoplane_positions = old_orthoplane_positions[0], old_orthoplane_positions[1], self.pos
                if self.axis == Axis.CORONAL:
                    new_orthoplane_positions = old_orthoplane_positions[0], self.pos, old_orthoplane_positions[2]
                if self.axis == Axis.SAGGITAL:
                    new_orthoplane_positions = self.pos, old_orthoplane_positions[1], old_orthoplane_positions[2]
                self.view.drawing.parent.set_parameters(orthoplane_positions=new_orthoplane_positions)
                self.slider_moved = False
            bounds = self.view.drawing.bounds()
            # We use these offsets to align the camera to the actual center of the orthoplanes
            x_offset, y_offset, z_offset = bounds.center()
            x_size, y_size, z_size = bounds.size()
            x_max, y_max, z_max = bounds.xyz_max
            cameraView = 2 * math.tan(0.5 * math.radians(self.view.camera.field_of_view))
            cameraDistance = 1.2
            x_apparent, y_apparent, z_apparent = [
                x_size * cameraDistance / cameraView
                , y_size * cameraDistance / cameraView
                , z_size * cameraDistance / cameraView
            ]
            # Set initial camera view
            # TODO: Why does this break if the volume style is switched to, say, surface and back to orthoplanes?
            # TODO: Make the initial calculation for the magic number (700) depend on the size of the DICOM file
            if self.axis == Axis.AXIAL:
                self.origin = self.view.drawing.position.origin() + [x_offset, y_offset, z_offset + z_apparent - self.camera_offset]
            elif self.axis == Axis.CORONAL:
                self.origin = self.view.drawing.position.origin() + [x_offset, -y_offset - (y_apparent - self.camera_offset), z_offset]
            else:
                self.origin = self.view.drawing.position.origin() + [x_offset + x_apparent - self.camera_offset, y_offset, z_offset]
            camera = self.view.camera
            camera.position = Place(axes=self.axes, origin=self.origin)
            loc = self.locations
            loc.bottom = .05 * height
            loc.top = .95 * height
            ratio = math.tan(0.5 * 30)
            #if self.moving:
            #    eye = self.view.win_coord(0, 0, 0)
            #    eye[2] = 0
            #    loc.eye = eye
            #elif ratio * width / 1.1 < .45 * height:
            #    loc.eye = array(
            #        [.05 / 1.1 * width, height / 2, 0],
            #        dtype=float32
            #        )
            #else:
            loc.eye = array(loc.eye)

            sr = self.segmentation_radius
            #v = array(
            #    [
            #        loc.eye + [-sr, -sr, 0],
            #        loc.eye + [-sr, sr, 0],
            #        loc.eye + [sr, sr, 0],
            #        loc.eye + [sr, -sr, 0],
            #    ], dtype=float32
            #)
            v, t = self._circle_geometry()
            vc = array([[255, 0, 0, 255]] * len(v), dtype=uint8)
            ps = self.view.render.pixel_scale()
            v *= ps
            #t = array(
            #    [
            #        [0, 1], [1, 2], [2, 3], [3, 0],  # eye box
            #    ], dtype=int32
            #)
            self.segmentation_overlay.set_geometry(v, None, t)
            self.segmentation_overlay.vertex_colors = vc

            self.view.prepare_scene_for_drawing()
            self.view._draw_scene(self.view.camera, [self.view.drawing])
            self.view.finalize_draw()
            # if has_string_marker:
            #     text = b"End SideView"
            #     string_marker.glStringMarkerGREMEDY(len(text), text)
            self.view.drawing.display = old_disp_val
            # self.view.drawing._rendering_options.orthoplanes_shown = (True, True, True)
        except Exception as e:
            # This line is here so you can set a breakpoint on it and figure out what's going wrong
            # because ChimeraX's interface will not tell you.
            pass
        finally:
            # Target opengl context back to main graphics window.
            self.main_view.render.use_shared_context(mvwin)
        self.view.render.done_current()

    def _circle_geometry(self):
        def mirror_points_8(x, y):
            return [(x, y), (y, x), (-x, y), (-y, x), (x, -y), (y, -x), (-x, -y), (-y, -x)]
        sr = self.segmentation_radius
        v = [None]*(8*(sr+1))
        for x in range(sr + 1):
            y = math.isqrt((sr * sr) - (x * x))
            points = mirror_points_8(x, y)
            v[x] = points[0]
            v[x + ((sr + 1) * 1)] = points[1]
            v[x + ((sr + 1) * 2)] = points[2]
            v[x + ((sr + 1) * 3)] = points[3]
            v[x + ((sr + 1) * 4)] = points[4]
            v[x + ((sr + 1) * 5)] = points[5]
            v[x + ((sr + 1) * 6)] = points[6]
            v[x + ((sr + 1) * 7)] = points[7]
        fv = [self.locations.eye + [vt[0], vt[1], 0] for vt in v]
        t = []
        for i in range(0, len(v)):
            t.append([i, i + 1])
        t[0][1] = 0
        t[-1][1] = 0
        fv = array(fv, dtype=float32)
        t = array(t, dtype=int32)
        return fv, t


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
            self.locations.eye = (event.position().x(), self.view.window_size[1] - event.position().y(), 0)
            self.view.camera.redraw_needed = True
        self.last_mouse_position = None

    def wheelEvent(self, event):
        modifier = event.modifiers()
        if modifier == Qt.KeyboardModifier.ShiftModifier:
            if event.angleDelta().x() < 0:
                self.segmentation_radius += 1
            elif event.angleDelta().x() > 0:
                self.segmentation_radius -= 1
        elif modifier == Qt.KeyboardModifier.NoModifier:
            old_offset = self.camera_offset
            if event.angleDelta().y() < 0:
                self.camera_offset -= 15
            elif event.angleDelta().y() > 0:
                self.camera_offset += 15
            diff_z = old_offset - self.camera_offset
            shift = self.view.camera.position.transform_vector((0, 0, diff_z * 4 * self.view.pixel_size()))
            self.view.translate(shift)
        self.view.camera.redraw_needed = True

    def mouseMoveEvent(self, event):  # noqa
        b = event.button() | event.buttons()
        # Level or segment
        if b == Qt.MouseButton.NoButton or b == Qt.MouseButton.LeftButton:
            self.locations.eye = (event.position().x(), self.view.window_size[1] - event.position().y(), 0)
            self.view.camera.redraw_needed = True
            return
        # Zoom
        if b & Qt.MouseButton.RightButton or b & Qt.MouseButton.MiddleButton:
            x, y = event.position().x(), event.position().y()
            if not self.last_mouse_position:
                dy = 0
            else:
                dy = y - self.last_mouse_position[1]
            psize = self.view.pixel_size()
            self.last_mouse_position = [x, y]
            self.camera_offset += (-dy * psize) * 3
            self.view.camera.redraw_needed = True

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
            self.orthoplane_positions = self.view.drawing._rendering_options.orthoplane_positions
            if self.axis == Axis.AXIAL:
                self.slider.setMaximum(max_z)
                self.slider.setValue(self.orthoplane_positions[2])
            if self.axis == Axis.CORONAL:
                self.slider.setRange(-max_y, 0)
                self.slider.setValue(-self.orthoplane_positions[1])
            if self.axis == Axis.SAGGITAL:
                self.slider.setMaximum(max_x)
                self.slider.setValue(self.orthoplane_positions[0])


class SegmentationOverlay(Drawing):
    def __init__(self, name):
        super().__init__(name)
        self.display_style = Drawing.Dot
        self.use_lighting = False

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
