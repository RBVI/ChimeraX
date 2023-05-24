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

from Qt.QtCore import Qt, QEvent
from Qt.QtCore import QSize
from Qt.QtWidgets import (
    QVBoxLayout, QHBoxLayout
    , QWidget, QSlider
)

from Qt.QtGui import QContextMenuEvent, QEnterEvent
from Qt.QtGui import QWindow, QSurface
from chimerax.geometry import Place
from chimerax.graphics.camera import ortho
from ..graphics import OrthoplaneView, OrthoCamera
from chimerax.graphics import MonoCamera, Drawing
from chimerax.graphics.opengl import GL
from chimerax.core.models import Surface
from chimerax.map import Volume, VolumeSurface, VolumeImage
from chimerax.ui.widgets import ModelMenu
from .label import Label
from enum import IntEnum

AXIAL_PLANE_MOVED = "axial plane moved"
CORONAL_PLANE_MOVED = "coronal plane moved"
SAGGITAL_PLANE_MOVED = "saggital plane moved"
# ORTHO_MODEL_CHANGED = "ortho model changed"

orthoplane_triggers = [
    AXIAL_PLANE_MOVED, CORONAL_PLANE_MOVED, SAGGITAL_PLANE_MOVED
]

class Direction(IntEnum):
    HORIZONTAL = 0
    VERTICAL = 1

class Axis(IntEnum):
    AXIAL = 2
    CORONAL = 1
    SAGGITAL = 0

    def __str__(self):
        return self.name.lower()

    @property
    def transform(self):
        if self.value == 2:
            return [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]
        elif self.value == 1:
            return [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
        else:
            return [[0, 1, 0], [0, 0, 1], [1, 0, 0]]

    @property
    def positive_direction(self):
        return [-1,1][self.value != 1]


class PlaneViewerManager:
    def __init__(self, session):
        self.session = session
        self.axes = {}

    def register(self, viewer):
        self.axes[viewer.axis] = viewer

    def update_location(self, viewer):
        if viewer.axis == Axis.AXIAL:
            self.axes[Axis.CORONAL].axial_index = viewer.axial_index
            self.axes[Axis.SAGGITAL].axial_index = viewer.axial_index
        if viewer.axis == Axis.CORONAL:
            self.axes[Axis.AXIAL].coronal_index = viewer.coronal_index
            self.axes[Axis.SAGGITAL].coronal_index = viewer.coronal_index
        if viewer.axis == Axis.SAGGITAL:
            self.axes[Axis.AXIAL].saggital_index = viewer.saggital_index
            self.axes[Axis.CORONAL].saggital_index = viewer.saggital_index

    def update_dimensions(self, dimensions):
        for axis in self.axes.values():
            axis.update_dimensions(dimensions)

     #def update_volume(self, viewer):
     #   if viewer.axis == Axis.AXIAL:
     #       self.axes[Axis.CORONAL].


class PlaneViewer(QWindow):

    def __init__(self, parent, manager, session, axis = Axis.AXIAL):
        QWindow.__init__(self)
        self.session = session
        self.manager = manager
        self.axis = axis
        self.manager.register(self)

        self.slider_moved = False
        self.last_mouse_position = None
        self.widget = QWidget.createWindowContainer(self, parent)
        self.setSurfaceType(QSurface.SurfaceType.OpenGLSurface)
        self.view = OrthoplaneView(Drawing("placeholder"), window_size = (0, 0), axis = self.axis)
        self.view.initialize_rendering(session.main_view.render.opengl_context)
        self.view.camera = OrthoCamera()
        self.view.camera.field_width = 500
        self.axes = axis.transform
        camera = self.view.camera
        camera.position = Place(origin = (0,0,0), axes = self.axes)
        self.view.background_color = (255, 255, 255, 255)
        self.main_view = session.main_view
        self.camera_offsets = [0, 0, 0]
        self._plane_indices = [0, 0, 0]
        self.label = Label(self.session, self.view, str(axis), str(axis), size=16, xpos=0, ypos=0)
        self.segmentation_slices = {}

        def _not_volume_surface(m):
            return not isinstance(m, VolumeSurface)

        self.model_menu = ModelMenu(
            self.session, parent, label = 'Model',
            model_types = [Volume, Surface],
            model_filter = _not_volume_surface,
            model_chosen_cb = self._surface_chosen
        )

        max_x, max_y, max_z = 2, 2, 2

        self.segmentation_overlay = SegmentationOverlay("seg_overlay", radius=10, thickness=3)
        self.horizontal_slice_overlay = OrthoplaneLocationOverlay("horiz_overlay", slice=10, direction=Direction.HORIZONTAL)
        self.vertical_slice_overlay = OrthoplaneLocationOverlay("vertical_overlay", slice=11)
        self.view.add_overlay(self.segmentation_overlay)
        self.view.add_overlay(self.horizontal_slice_overlay)
        self.view.add_overlay(self.vertical_slice_overlay)
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
        self.widget.setMinimumSize(QSize(20, 20))

        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(0)
        button_layout.addStretch(1)
        button_layout.addWidget(self.model_menu.frame)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.container = QWidget(parent)
        container_layout = QVBoxLayout()
        container_layout.addLayout(button_layout)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)
        container_layout.addWidget(self.widget, stretch=1)
        container_layout.addWidget(self.slider)
        self.container.setLayout(container_layout)

        layout.addWidget(self.container, stretch=1)
        self.container.setLayout(layout)

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
            if not self.view.drawing.parent.data.dicom_data.inferior_to_superior:
                diff = -diff
        self.camera_offsets[self.axis] -= diff * self.view.drawing.parent.data.step[self.axis]
        self._plane_indices[self.axis] = self.pos
        self.manager.update_location(self)
        self.view.camera.redraw_needed = True

    def close(self):
        self.session.triggers.remove_handler(self.handler)
        # TODO: why does this call make it crash?
        # self.setParent(None)
        self.label.delete()
        del self.label
        self.view.delete()
        QWindow.destroy(self)

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
            # TODO: Turn on when overlay calculations are correct
            # self.view.background_color = self.main_view.background_color

            old_disp_val = self.view.drawing.display
            self.scale = mvwin.opengl_context.pixel_scale()
            if not old_disp_val:
                self.view.drawing.display = True
            # TODO: If the user selects 'surface' then 'orthoplanes' in the volume viewer we should
            # override the default plane locations somehow
            if self.slider_moved:
                self.view.drawing.parent.set_parameters(orthoplane_positions=tuple(self._plane_indices))
                self.view.drawing.parent.update_drawings()
                self.slider_moved = False
            bounds = self.view.drawing.bounds()
            # We use these offsets to align the camera to the actual center of the orthoplanes
            model_center_offsets = bounds.center()
            psize = self.view.pixel_size()
            axis_sizes = (bounds.size() / psize)[::-1]

            # TODO: Use the difference between the camera center and the object offsets to compensate for moving the
            # camera around
            if self.axis == Axis.SAGGITAL:
                x_offset, y_offset = self.camera_offsets[Axis.CORONAL] * self.scale / psize, self.camera_offsets[Axis.AXIAL] * self.scale / psize
                self.horizontal_slice_overlay.bottom = 0.5 * self.scale * ((width + x_offset) - axis_sizes[Axis.AXIAL])
                self.horizontal_slice_overlay.top =    0.5 * self.scale * ((width + x_offset) + axis_sizes[Axis.AXIAL])
                self.horizontal_slice_overlay.offset = 0.5 * self.scale * ((height + y_offset) - axis_sizes[Axis.SAGGITAL])
                self.horizontal_slice_overlay.tick_thickness = axis_sizes[Axis.SAGGITAL] / (self.dimensions[Axis.AXIAL] / self.scale)

                self.vertical_slice_overlay.bottom = 0.5 * self.scale * ((height + y_offset) - axis_sizes[Axis.SAGGITAL])
                self.vertical_slice_overlay.top =    0.5 * self.scale * ((height + y_offset) + axis_sizes[Axis.SAGGITAL])
                self.vertical_slice_overlay.offset = 0.5 * self.scale * ((width + x_offset) - axis_sizes[Axis.CORONAL])
                self.vertical_slice_overlay.tick_thickness = axis_sizes[Axis.CORONAL] / (self.dimensions[Axis.CORONAL] / self.scale)

            elif self.axis == Axis.CORONAL:
                x_offset, y_offset = self.camera_offsets[Axis.SAGGITAL] * self.scale / psize, self.camera_offsets[Axis.AXIAL] * self.scale / psize
                self.horizontal_slice_overlay.bottom = 0.5 * self.scale * ((width + x_offset) - axis_sizes[Axis.CORONAL])
                self.horizontal_slice_overlay.top =    0.5 * self.scale * ((width + x_offset) + axis_sizes[Axis.CORONAL])
                self.horizontal_slice_overlay.offset = 0.5 * self.scale * ((height + y_offset) - axis_sizes[Axis.SAGGITAL])
                self.horizontal_slice_overlay.tick_thickness = axis_sizes[Axis.SAGGITAL] / (self.dimensions[Axis.AXIAL] / self.scale)

                self.vertical_slice_overlay.bottom = 0.5 * self.scale * ((height + y_offset) - axis_sizes[Axis.SAGGITAL])
                self.vertical_slice_overlay.top =    0.5 * self.scale * ((height + y_offset) + axis_sizes[Axis.SAGGITAL])
                self.vertical_slice_overlay.offset = 0.5 * self.scale * ((width + x_offset) - axis_sizes[Axis.AXIAL])
                self.vertical_slice_overlay.tick_thickness = axis_sizes[Axis.AXIAL] / (self.dimensions[Axis.SAGGITAL] / self.scale)

            else:
                x_offset, y_offset = self.camera_offsets[Axis.SAGGITAL] * self.scale / psize, self.camera_offsets[Axis.CORONAL] * self.scale / psize
                self.horizontal_slice_overlay.bottom = 0.5 * self.scale * ((width - x_offset) - axis_sizes[Axis.AXIAL])
                self.horizontal_slice_overlay.top =    0.5 * self.scale * ((width - x_offset) + axis_sizes[Axis.AXIAL])
                self.horizontal_slice_overlay.offset = 0.5 * self.scale * ((height - y_offset) + axis_sizes[Axis.AXIAL])
                self.horizontal_slice_overlay.tick_thickness = axis_sizes[Axis.AXIAL] / (self.dimensions[Axis.CORONAL] / self.scale)

                self.vertical_slice_overlay.bottom = 0.5 * self.scale * ((height - y_offset) - axis_sizes[Axis.CORONAL])
                self.vertical_slice_overlay.top =    0.5 * self.scale * ((height - y_offset) + axis_sizes[Axis.CORONAL])
                self.vertical_slice_overlay.offset = 0.5 * self.scale * ((width - x_offset) + axis_sizes[Axis.CORONAL])
                self.vertical_slice_overlay.tick_thickness = axis_sizes[Axis.CORONAL] / (self.dimensions[Axis.SAGGITAL] / self.scale)

            test_c_offsets = [0, 0, 0]
            self.origin = self.view.drawing.position.origin() + model_center_offsets - self.camera_offsets + test_c_offsets
            camera = self.view.camera
            camera.position = Place(axes=self.axes, origin=self.origin)
            self.segmentation_overlay.update()
            self.horizontal_slice_overlay.update()
            self.vertical_slice_overlay.update()

            self.view.prepare_scene_for_drawing()
            self.view._draw_scene(self.view.camera, [self.view.drawing])
            self.view.finalize_draw()
            self.view.drawing.display = old_disp_val
        except Exception as e: # noqa
            # This line is here so you can set a breakpoint on it and figure out what's going wrong
            # because ChimeraX's interface will not tell you.
            pass
        finally:
            # Target opengl context back to main graphics window.
            self.main_view.render.use_shared_context(mvwin)
        self.view.render.done_current()

    def event(self, event):
        if event.type() == QEvent.Type.Enter:
            self.enterEvent()
        if event.type() == QEvent.Type.Leave:
            self.leaveEvent()
        return QWindow.event(self, event)

    def enableSegmentationOverlay(self):
        self.segmentation_overlay.display = True

    def disableSegmentationOverlay(self):
        self.segmentation_overlay.display = False

    def enterEvent(self):
        self.enableSegmentationOverlay()

    def leaveEvent(self):
        self.disableSegmentationOverlay()

    def mousePressEvent(self, event):  # noqa
        b = event.button() | event.buttons()
        if b & Qt.MouseButton.RightButton:
            # p = event.position() if hasattr(event, 'position') else event.pos()  # PyQt6 / PyQt5
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
            self.segmentation_overlay.center = (self.scale * event.position().x(), self.scale * (self.view.window_size[1] - event.position().y()), 0)
            self.segmentation_overlay.update()
            self.view.camera.redraw_needed = True
        self.last_mouse_position = None

    def wheelEvent(self, event):
        modifier = event.modifiers()
        delta = event.angleDelta()
        x_dir, y_dir = np.sign(delta.x()), np.sign(delta.y())
        if modifier == Qt.KeyboardModifier.ShiftModifier:
            self.segmentation_overlay.radius += 1 * x_dir
        elif modifier == Qt.KeyboardModifier.NoModifier:
            self.camera_offsets[self.axis] += 15 * y_dir * self.axis.positive_direction
            self.view.camera.field_width += 1 * y_dir
        self.view.camera.redraw_needed = True

    def mouseMoveEvent(self, event):  # noqa
        b = event.button() | event.buttons()
        # Level or segment
        if b == Qt.MouseButton.NoButton or b == Qt.MouseButton.LeftButton:
            pos = event.position()
            self.segmentation_overlay.center = (self.scale * pos.x(), self.scale * (self.view.window_size[1] - pos.y()), 0)
            self.segmentation_overlay.update()
            self.view.camera.redraw_needed = True
            return
        # Zoom / Dolly
        if b & Qt.MouseButton.RightButton:
            pos = event.position()
            x, y = pos.x(), pos.y()
            if not self.last_mouse_position:
                dy = 0
            else:
                dy = y - self.last_mouse_position[1]
            psize = self.view.pixel_size()
            self.last_mouse_position = [x, y]
            self.view.camera.field_width -= 1 # offsets[self.axis] += (-dy * psize) * 3 * self.axis.positive_direction
            self.view.camera.redraw_needed = True
        # Truck & Pedestal
        if b & Qt.MouseButton.MiddleButton:
            pos = event.position()
            x, y = pos.x(), pos.y()
            p_size = self.view.pixel_size()
            if not self.last_mouse_position:
                dx = 0
                dy = 0
            else:
                dx = (x - self.last_mouse_position[0]) * p_size
                dy = (y - self.last_mouse_position[1]) * p_size
            self.last_mouse_position = [x, y]
            x, y, z = self.camera_offsets
            if self.axis == Axis.AXIAL:
                self.camera_offsets = [x - dx, y + dy, z]
            if self.axis == Axis.CORONAL:
                self.camera_offsets = [x + dx, y, z - dy]
            if self.axis == Axis.SAGGITAL:
                self.camera_offsets = [x, y + dx, z - dy]

    def keyPressEvent(self, event):  # noqa
        return self.session.ui.forward_keystroke(event)

    def update_dimensions(self, dimensions):
        self.dimensions = dimensions

    @property
    def axial_index(self):
        return self._plane_indices[Axis.AXIAL]

    @property
    def coronal_index(self):
        return self._plane_indices[Axis.CORONAL]

    @property
    def saggital_index(self):
        return self._plane_indices[Axis.SAGGITAL]

    @axial_index.setter
    def axial_index(self, index):
        self._plane_indices[Axis.AXIAL] = index
        # TODO: Have the overlay get this itself?
        if self.axis == Axis.CORONAL:
            self.horizontal_slice_overlay.slice = index
        if self.axis == Axis.SAGGITAL:
            self.horizontal_slice_overlay.slice = index

    @coronal_index.setter
    def coronal_index(self, index):
        self._plane_indices[Axis.CORONAL] = index
        # TODO: Have the overlay get this itself?
        if self.axis == Axis.AXIAL:
            self.horizontal_slice_overlay.slice = -index
        if self.axis == Axis.SAGGITAL:
            self.vertical_slice_overlay.slice = index

    @saggital_index.setter
    def saggital_index(self, index):
        self._plane_indices[Axis.SAGGITAL] = index
        # TODO: Have the overlay get this itself?
        if self.axis == Axis.AXIAL:
            self.vertical_slice_overlay.slice = -index
        if self.axis == Axis.CORONAL:
            self.vertical_slice_overlay.slice = index

    def _surface_chosen(self, *args):
        new_drawing = None
        for d in self.model_menu.value._child_drawings:
            if type(d) is VolumeImage:
                new_drawing = d
        #self.manager.update_drawing(self.model_menu.value)
        if new_drawing is not None:
            self.view.drawing = new_drawing
            self.set_label_text(d.parent.name)
            max_x, max_y, max_z = max_slider_vals = self.view.drawing._region[1]
            self.manager.update_dimensions([max_x, max_y, max_z])
            orthoplane_positions = self.view.drawing._rendering_options.orthoplane_positions
            self._plane_indices = list(orthoplane_positions)
            if self.axis == Axis.AXIAL:
                self.slider.setMaximum(max_z)
            if self.axis == Axis.CORONAL:
                self.slider.setRange(-max_y, 0)
            if self.axis == Axis.SAGGITAL:
                self.slider.setMaximum(max_x)
            self.slider.setValue(orthoplane_positions[self.axis] * self.axis.positive_direction)
            self.pos = orthoplane_positions[self.axis]
            self._plane_indices[self.axis] = self.pos
            self.manager.update_location(self)
            self.view.camera.redraw_needed = True

    def set_label_text(self, text):
        self.label.text = text
        self.label.update_drawing()

    def set_volume(self, new_volume):
        if self.model_menu.value == new_volume:
            return
        ...


class OrthoplaneLocationOverlay(Drawing):
    def __init__(self, name, slice, direction = Direction.VERTICAL):
        super().__init__(name)
        self.max_line_width = max(GL.glGetIntegerv(GL.GL_LINE_WIDTH_RANGE)[1], 1)
        self.display_style = Drawing.Mesh
        self.use_lighting = False
        self.direction = direction
        self._slice = slice
        self.bottom = 0
        self.offset = 0
        self.top = 0
        self.tick_thickness = 1

    def draw(self, renderer, draw_pass):
        r = renderer
        ww, wh = r.render_size()
        projection = ortho(0, ww, 0, wh, -1, 1)
        r.set_projection_matrix(projection)
        Drawing.draw(self, renderer, draw_pass)
        r.set_projection_matrix((
            (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)
        ))

    @property
    def slice(self):
        return self._slice

    @slice.setter
    def slice(self, slice):
        self._slice = slice

    def update(self):
        vc, v, _, t = self._geometry()
        self.set_geometry(v, _, t)
        self.vertex_colors = vc

    # TODO: Depend on the slice location
    def _geometry(self):
        if self.direction == Direction.VERTICAL:
            ofs = (self.slice * self.tick_thickness) + self.offset
            v = [[ofs-1, self.bottom, 0], [ofs-1, self.top, 0], [ofs, self.top, 0], [ofs, self.bottom, 0], [ofs+1, self.bottom, 0], [ofs+1, self.top, 0]]
            t = [[0, 1], [1,2], [2,3], [3,4], [4,5], [5,0]]
            v = array(v, dtype=float32)
            t = array(t, dtype=int32)
            c = array([[255, 0, 0, 255]] * len(v), dtype=uint8)
        else:
            ofs = (self.slice * self.tick_thickness) + self.offset
            v = [[self.bottom, ofs, 0], [self.top, ofs, 0], [self.top ,ofs + 1,0], [self.bottom,ofs+1,0]]
            t = [[0, 1], [1, 2], [2, 3], [3, 0]]
            v = array(v, dtype=float32)
            t = array(t, dtype=int32)
            c = array([[255, 0, 0, 255]] * len(v), dtype=uint8)
        return c, v, None, t


class SegmentationOverlay(Drawing):
    def __init__(self, name, radius, thickness):
        super().__init__(name)
        self.max_point_size = GL.glGetIntegerv(GL.GL_POINT_SIZE_RANGE)[1]
        self.display_style = Drawing.Dot
        self.use_lighting = False
        self._radius = radius
        self._center = [0, 0, 0]
        self._thickness = thickness

    def draw(self, renderer, draw_pass):
        GL.glPointSize(min(self.max_point_size, self.thickness))
        r = renderer
        ww, wh = r.render_size()
        projection = ortho(0, ww, 0, wh, -1, 1)
        r.set_projection_matrix(projection)
        Drawing.draw(self, renderer, draw_pass)
        r.set_projection_matrix(
            ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0),
             (0, 0, 0, 1))
            )
        GL.glPointSize(1)

    @property
    def thickness(self):
        return self._thickness

    @thickness.setter
    def thickness(self, thickness):
        if self.max_point_size > thickness > 0:
            self._thickness = thickness
        else:
            raise ValueError("Thickness exceeds OpenGL limit")

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
