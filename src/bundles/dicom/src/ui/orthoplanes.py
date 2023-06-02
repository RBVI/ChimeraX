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
import numpy as np

from Qt.QtCore import Qt, QEvent, QSize
from Qt.QtGui import QContextMenuEvent, QWindow, QSurface
from Qt.QtWidgets import QVBoxLayout, QHBoxLayout, QWidget, QSlider

from chimerax.core.models import Surface
from chimerax.geometry import Place, translation
from chimerax.graphics import Drawing
from chimerax.map import Volume, VolumeSurface, VolumeImage
from chimerax.ui.widgets import ModelMenu

from ..graphics import OrthoplaneView, OrthoCamera, SegmentationOverlay, OrthoplaneLocationOverlay
from ..types import Direction, Axis
from .label import Label

class PlaneViewerManager:
    def __init__(self, session):
        self.session = session
        self.axes = {}

    def register(self, viewer):
        self.axes[viewer.axis] = viewer

    def update_location(self, viewer):
        if viewer.axis == Axis.AXIAL:
            if Axis.CORONAL in self.axes:
                self.axes[Axis.CORONAL].axial_index = viewer.axial_index
            if Axis.SAGGITAL in self.axes:
                self.axes[Axis.SAGGITAL].axial_index = viewer.axial_index
        if viewer.axis == Axis.CORONAL:
            if Axis.AXIAL in self.axes:
                self.axes[Axis.AXIAL].coronal_index = viewer.coronal_index
            if Axis.SAGGITAL in self.axes:
                self.axes[Axis.SAGGITAL].coronal_index = viewer.coronal_index
        if viewer.axis == Axis.SAGGITAL:
            if Axis.AXIAL in self.axes:
                self.axes[Axis.AXIAL].saggital_index = viewer.saggital_index
            if Axis.CORONAL in self.axes:
                self.axes[Axis.CORONAL].saggital_index = viewer.saggital_index

    def update_dimensions(self, dimensions):
        for axis in self.axes.values():
            axis.update_dimensions(dimensions)

    def register_segmentation_tool(self, tool):
        for viewer in self.axes.values():
            viewer.segmentation_tool = tool

    def clear_segmentation_tool(self):
        for viewer in self.axes.values():
            viewer.segmentation_tool = None

    def show_guidelines(self):
        for viewer in self.axes.values():
            viewer.set_guideline_visibility(True)

    def hide_guidelines(self):
        for viewer in self.axes.values():
            viewer.set_guideline_visibility(False)

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
        self._segmentation_tool = None
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
        self.horizontal_slice_overlay.display = False
        self.horizontal_slice_overlay.display = False
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

    @property
    def segmentation_tool(self):
        return self._segmentation_tool

    @segmentation_tool.setter
    def segmentation_tool(self, tool):
        self._segmentation_tool = tool
        self._segmentation_tool.segmentation_cursors[self.axis].radius = self.segmentation_overlay.radius
        # TODO:
        # Set the segmentation pucks' locations based on the current slice location
        # self._segmentation_tool.segmentation_cursors[self.axis].
        self._segmentation_tool.model_menu.value = self.model_menu.value
        # Set their radii to the current selected models' thickness
        # Synchronize the tool's model menu value to our model menu value

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
        # TODO: Set the segmentation drawing's position to coincide with the new slice
        if self.segmentation_tool:
            offset_from_origin = self.pos * self.view.drawing.parent.data.step[self.axis]
            old_origin = self.segmentation_tool.segmentation_cursors[self.axis].origin
            origin = old_origin
            origin[self.axis] = self.view.drawing.parent.data.dicom_data.origin()[self.axis] + offset_from_origin
            self.segmentation_tool.segmentation_cursors[self.axis].origin = old_origin
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
            model_center_offsets = self.view.drawing.bounds().center()
            self.calculateSliceOverlays()
            test_c_offsets = [[15, 0, 0], [0,15,0],[0,0,15]][self.axis]
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


    def calculateSliceOverlays(self):
        width, height = self.view.window_size
        bounds = self.view.drawing.bounds()
        psize = self.view.pixel_size()
        axis_sizes = (bounds.size() / psize)[::-1]
        x_offset, y_offset = self.cameraSpaceDrawingOffsets()
        if self.axis == Axis.SAGGITAL:
            self.horizontal_slice_overlay.bottom = 0.5 * self.scale * ((width + x_offset) - axis_sizes[Axis.AXIAL])
            self.horizontal_slice_overlay.top =    0.5 * self.scale * ((width + x_offset) + axis_sizes[Axis.AXIAL])
            self.horizontal_slice_overlay.offset = 0.5 * self.scale * ((height + y_offset) - axis_sizes[Axis.SAGGITAL])
            self.horizontal_slice_overlay.tick_thickness = axis_sizes[Axis.SAGGITAL] / (self.dimensions[Axis.AXIAL] / self.scale)

            self.vertical_slice_overlay.bottom = 0.5 * self.scale * ((height + y_offset) - axis_sizes[Axis.SAGGITAL])
            self.vertical_slice_overlay.top =    0.5 * self.scale * ((height + y_offset) + axis_sizes[Axis.SAGGITAL])
            self.vertical_slice_overlay.offset = 0.5 * self.scale * ((width + x_offset) - axis_sizes[Axis.CORONAL])
            self.vertical_slice_overlay.tick_thickness = axis_sizes[Axis.CORONAL] / (self.dimensions[Axis.CORONAL] / self.scale)

        elif self.axis == Axis.CORONAL:
            self.horizontal_slice_overlay.bottom = 0.5 * self.scale * ((width + x_offset) - axis_sizes[Axis.CORONAL])
            self.horizontal_slice_overlay.top =    0.5 * self.scale * ((width + x_offset) + axis_sizes[Axis.CORONAL])
            self.horizontal_slice_overlay.offset = 0.5 * self.scale * ((height + y_offset) - axis_sizes[Axis.SAGGITAL])
            self.horizontal_slice_overlay.tick_thickness = axis_sizes[Axis.SAGGITAL] / (self.dimensions[Axis.AXIAL] / self.scale)

            self.vertical_slice_overlay.bottom = 0.5 * self.scale * ((height + y_offset) - axis_sizes[Axis.SAGGITAL])
            self.vertical_slice_overlay.top =    0.5 * self.scale * ((height + y_offset) + axis_sizes[Axis.SAGGITAL])
            self.vertical_slice_overlay.offset = 0.5 * self.scale * ((width + x_offset) - axis_sizes[Axis.AXIAL])
            self.vertical_slice_overlay.tick_thickness = axis_sizes[Axis.AXIAL] / (self.dimensions[Axis.SAGGITAL] / self.scale)

        else:
            self.horizontal_slice_overlay.bottom = 0.5 * self.scale * ((width - x_offset) - axis_sizes[Axis.AXIAL])
            self.horizontal_slice_overlay.top =    0.5 * self.scale * ((width - x_offset) + axis_sizes[Axis.AXIAL])
            self.horizontal_slice_overlay.offset = 0.5 * self.scale * ((height - y_offset) + axis_sizes[Axis.AXIAL])
            self.horizontal_slice_overlay.tick_thickness = axis_sizes[Axis.AXIAL] / (self.dimensions[Axis.CORONAL] / self.scale)

            self.vertical_slice_overlay.bottom = 0.5 * self.scale * ((height - y_offset) - axis_sizes[Axis.CORONAL])
            self.vertical_slice_overlay.top =    0.5 * self.scale * ((height - y_offset) + axis_sizes[Axis.CORONAL])
            self.vertical_slice_overlay.offset = 0.5 * self.scale * ((width - x_offset) + axis_sizes[Axis.CORONAL])
            self.vertical_slice_overlay.tick_thickness = axis_sizes[Axis.CORONAL] / (self.dimensions[Axis.SAGGITAL] / self.scale)

    def camera_space_drawing_bounds(self):
        top, bottom, left, right = (
            self.horizontal_slice_overlay.top
            , self.horizontal_slice_overlay.bottom
            , self.vertical_slice_overlay.bottom
            , self.vertical_slice_overlay.top
        )
        return top, bottom, left, right

    def event(self, event):
        if event.type() == QEvent.Type.Enter:
            self.enterEvent()
        if event.type() == QEvent.Type.Leave:
            self.leaveEvent()
        return QWindow.event(self, event)

    def set_guideline_visibility(self, visibility: bool):
        self.horizontal_slice_overlay.display = visibility
        self.horizontal_slice_overlay.display = visibility

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
            self.segmentation_overlay.radius += self.scale * 0.5 * x_dir
            if self.segmentation_tool:
                self.segmentation_tool.segmentation_cursors[self.axis].radius += 1 * x_dir
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
            # TODO: Take the center of the segmentation overlay and map it to the location on the shown slice
            if self.segmentation_tool:
                top, bottom, left, right = self.camera_space_drawing_bounds()
                if not left <= pos.x() <= right and bottom <= pos.y() <= top:
                    return
                old_origin = self.segmentation_tool.segmentation_cursors[self.axis].origin
                offset_left = pos.x() - bottom
                offset_bottom = pos.y() - bottom
                drawing_origin = self.view.drawing.parent.data.dicom_data.origin()
                origin = old_origin
                if self.axis == Axis.AXIAL:
                    origin[0], origin[1] = drawing_origin[0] + pos.x(), drawing_origin[1] + pos.y()
                if self.axis == Axis.CORONAL:
                    origin[0], origin[2] = drawing_origin[0] + pos.x(), drawing_origin[2] + pos.y()
                if self.axis == Axis.SAGGITAL:
                    origin[1], origin[2] = drawing_origin[1] - pos.x(), drawing_origin[2] - pos.y()
                self.segmentation_tool.segmentation_cursors[self.axis].origin = origin
            self.view.camera.redraw_needed = True
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
