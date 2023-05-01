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
        self.session = session
        self.axis = axis
        self.overlays = []
        p2d = list(self.session.models)[-1]
        self.widget = QWidget.createWindowContainer(self, parent)
        self.setSurfaceType(QSurface.SurfaceType.OpenGLSurface)
        self.view = OrthoplaneView(p2d, window_size = (0, 0), axis = self.axis.value)
        self.view.initialize_rendering(session.main_view.render.opengl_context)
        # TODO: from chimerax.graphics.camera import OrthographicCamera
        self.view.camera = MonoCamera()
        self.view.background_color = (255, 255, 255, 255)
        self.panel = panel
        self.main_view = session.main_view
        self.moving = self.ON_NOTHING
        self.camera_offset = 0
        self.label = Label(self.session, self.view, str(axis), str(axis), size=16, xpos=0, ypos=0)

        max_slider_vals = self.view.drawing._region[1]
        max_x, max_y, max_z = max_slider_vals

        self.slider = QSlider(Qt.Orientation.Horizontal, parent)
        self.orthoplane_positions = self.view.drawing._rendering_options.orthoplane_positions
        if axis == Axis.AXIAL:
            self.slider.setMaximum(max_z)
            self.slider.setValue(self.orthoplane_positions[2])
        if axis == Axis.CORONAL:
            self.slider.setMaximum(max_y)
            self.slider.setValue(self.orthoplane_positions[1])
        if axis == Axis.SAGGITAL:
            self.slider.setMaximum(max_x)
            self.slider.setValue(self.orthoplane_positions[0])
        self.old_pos = 0
        self.pos = self.slider.value()
        self.slider.sliderMoved.connect(self._on_slider_moved)

        class _PixelLocations:
            pass

        self.locations = loc = _PixelLocations()
        loc.eye = 0, 0, 0   # x, y coordinates of eye
        loc.near = 0        # X coordinate of near plane
        loc.far = 0         # Y coordinate of near plane
        loc.bottom = 0      # bottom of clipping planes
        loc.top = 0         # top of clipping planes
        loc.far_bottom = 0  # right clip intersect far
        loc.far_top = 0     # left clip intersect far

        self.handler = session.triggers.add_handler('frame drawn', self._redraw)
        from Qt.QtCore import QSize
        self.widget.setMinimumSize(QSize(20, 20))

    def _on_slider_moved(self):
        self.old_pos = self.pos
        self.pos = self.slider.sliderPosition()
        diff = self.pos - self.old_pos
        if self.axis == Axis.AXIAL:
            self.camera_offset -= diff * self.view.drawing.parent.data.step[2]
        if self.axis == Axis.CORONAL:
            self.camera_offset += diff * self.view.drawing.parent.data.step[1]
        if self.axis == Axis.SAGGITAL:
            self.camera_offset -= diff * self.view.drawing.parent.data.step[0]
        self._redraw()

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
            new_orthoplane_positions = old_orthoplane_positions = self.view.drawing._rendering_options.orthoplane_positions
            if self.axis == Axis.AXIAL:
                new_orthoplane_positions = old_orthoplane_positions[0], old_orthoplane_positions[1], self.pos
            if self.axis == Axis.CORONAL:
                new_orthoplane_positions = old_orthoplane_positions[0], self.pos, old_orthoplane_positions[2]
            if self.axis == Axis.SAGGITAL:
                new_orthoplane_positions = self.pos, old_orthoplane_positions[1], old_orthoplane_positions[2]
            self.view.drawing.parent.set_parameters(orthoplane_positions=new_orthoplane_positions)
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
                axes = [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]
            elif self.axis == Axis.CORONAL:
                self.origin = self.view.drawing.position.origin() + [x_offset, -y_offset - (y_apparent - self.camera_offset), z_offset]
                axes = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
            else:
                self.origin = self.view.drawing.position.origin() + [x_offset + x_apparent - self.camera_offset, y_offset, z_offset]
                axes = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
            self.x = self.origin[0]
            self.y = self.origin[1]
            camera = self.view.camera
            camera_pos = Place(axes=axes, origin=self.origin)
            camera.position = camera_pos
            if not old_disp_val:
                self.view.drawing.display = True
            self.view.prepare_scene_for_drawing()
            self.view._draw_scene(self.view.camera, [self.view.drawing])
            self.view.finalize_draw()
            # if has_string_marker:
            #     text = b"End SideView"
            #     string_marker.glStringMarkerGREMEDY(len(text), text)
            self.view.drawing.display = old_disp_val
            # self.view.drawing._rendering_options.orthoplanes_shown = (True, True, True)
        finally:
            # Target opengl context back to main graphics window.
            self.main_view.render.use_shared_context(mvwin)
        self.view.render.done_current()

    def mousePressEvent(self, event):  # noqa
        b = event.button() | event.buttons()
        if b & Qt.MouseButton.RightButton:
            from Qt.QtGui import QContextMenuEvent
            e = QContextMenuEvent(QContextMenuEvent.Mouse, event.pos())
            self.widget.parent().parent().contextMenuEvent(e)
            return
        if b & Qt.MouseButton.MiddleButton:
            return
        if b & Qt.MouseButton.LeftButton:
            p = event.position() if hasattr(event, 'position') else event.pos()  # PyQt6 / PyQt5
            x, y = p.x(), p.y()
            eye_x, eye_y = self.locations.eye[0:2]
            near = self.locations.near
            far = self.locations.far
            es = self.EyeSize
            if eye_x - es <= x <= eye_x + es and eye_y - es <= y <= eye_y + es:
                self.moving = self.ON_EYE
            elif near - es <= x <= near + es:
                self.moving = self.ON_NEAR
            elif far - es <= x <= far + es:
                self.moving = self.ON_FAR
            else:
                return
            self.x, self.y = x, y
            return

    def mouseReleaseEvent(self, event):  # noqa
        if not self.moving:
            return
        from Qt.QtCore import Qt
        b = event.button() | event.buttons()
        if b & Qt.LeftButton:
            self.moving = self.ON_NOTHING
            self.exposeEvent(None)

    def wheelEvent(self, event):
        old_offset = self.camera_offset
        if event.angleDelta().y() < 0:
            self.camera_offset -= 10
        elif event.angleDelta().y() > 0:
            self.camera_offset += 10
        diff_z = old_offset - self.camera_offset
        shift = self.view.camera.position.transform_vector((0, 0, diff_z * 4 * self.view.pixel_size()))
        self.view.translate(shift)
        self.render()

    def mouseMoveEvent(self, event):  # noqa
        b = event.button() | event.buttons()
        # Level or segment
        if b & Qt.MouseButton.LeftButton:
            return
        # Zoom
        if b & Qt.MouseButton.RightButton:
            p = event.position()
            y = p.y()
            diff_y = y - self.y
            self.y = y
            psize = self.view.pixel_size()
            shift = self.view.camera.position.transform_vector((0, 0, -diff_y * psize))
            self.view.translate(shift)
        # Translate
        if b & Qt.MouseButton.MiddleButton:
            p = event.position()
            x, y = p.x(), p.y()
            diff_x = x - self.x
            diff_y = y - self.y
            self.x, self.y = x, y
            psize = self.view.pixel_size()
            shift = self.view.camera.position.transform_vector((diff_x * psize, -diff_y * psize, 0))
            self.view.translate(shift)
        self.render()

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



class SegmentationOverlay(Drawing):
    def draw(renderer, draw_pass):
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
