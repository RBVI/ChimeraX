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

    def toggle_guidelines(self):
        for viewer in self.axes.values():
            viewer.setGuidelineVisibility(not viewer.guidelines_visible)

    def show_guidelines(self):
        for viewer in self.axes.values():
            viewer.setGuidelineVisibility(True)

    def hide_guidelines(self):
        for viewer in self.axes.values():
            viewer.setGuidelineVisibility(False)

     #def update_volume(self, viewer):
     #   if viewer.axis == Axis.AXIAL:
     #       self.axes[Axis.CORONAL].


class PlaneViewer(QWindow):

    def __init__(self, parent, manager, session, axis = Axis.AXIAL):
        QWindow.__init__(self)
        self.parent = parent
        self.manager = manager
        self.session = session
        self.axis = axis
        self.axes = axis.transform
        self._segmentation_tool = None
        self.manager.register(self)

        self.last_mouse_position = None

        self.widget = QWidget.createWindowContainer(self, parent)
        self.setSurfaceType(QSurface.SurfaceType.OpenGLSurface)
        self.view = OrthoplaneView(Drawing("placeholder"), window_size = (0, 0), axis = self.axis)
        self.view.initialize_rendering(session.main_view.render.opengl_context)
        self.view.camera = OrthoCamera()
        self.view.camera.field_width = 500

        camera = self.view.camera
        camera.position = Place(origin = (0,0,0), axes = self.axes)

        # TODO: Set to main_view background color and update it in render loop
        self.view.background_color = (255, 255, 255, 255)

        self.main_view = session.main_view

        self.camera_offsets = [0, 0, 0]
        self._plane_indices = [0, 0, 0]

        self.label = Label(self.session, self.view, str(axis), str(axis), size=16, xpos=0, ypos=0)

        def _not_volume_surface(m):
            return not isinstance(m, VolumeSurface)

        self.model_menu = ModelMenu(
            self.session, parent, label = 'Model',
            model_types = [Volume, Surface],
            model_filter = _not_volume_surface,
            model_chosen_cb = self._surface_chosen
        )

        self.segmentation_overlay = SegmentationOverlay("seg_overlay", radius=10, thickness=3)
        self.horizontal_slice_overlay = OrthoplaneLocationOverlay("horiz_overlay", slice=10, direction=Direction.HORIZONTAL)
        self.vertical_slice_overlay = OrthoplaneLocationOverlay("vertical_overlay", slice=11)
        self.horizontal_slice_overlay.display = False
        self.vertical_slice_overlay.display = False
        self.segmentation_overlay.display = False
        self.segmentation_cursor_enabled = False
        self.view.add_overlay(self.segmentation_overlay)
        self.view.add_overlay(self.horizontal_slice_overlay)
        self.view.add_overlay(self.vertical_slice_overlay)

        # Used to move the camera when slices are moved
        self.old_pos = 0
        self.pos = 0

        self.slider = QSlider(Qt.Orientation.Horizontal, parent)
        max_x, max_y, max_z = 2, 2, 2
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
        self.slider_moved = False

        self.handler = session.triggers.add_handler('frame drawn', self._redraw)
        self.widget.setMinimumSize(QSize(20, 20))

        self.button_container = QWidget()
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(0)
        button_layout.addStretch(1)
        button_layout.addWidget(self.model_menu.frame)
        self.button_container.setLayout(button_layout)

        self.container = QWidget(parent)
        container_layout = QVBoxLayout()
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)
        container_layout.addWidget(self.button_container)
        container_layout.addWidget(self.widget, stretch=1)
        container_layout.addWidget(self.slider)
        self.container.setLayout(container_layout)

        self.context_menu = None
        self.context_menu_coords = None
        self.mouse_moved_during_right_click = False

    @property
    def segmentation_tool(self):
        return self._segmentation_tool

    @segmentation_tool.setter
    def segmentation_tool(self, tool):
        self._segmentation_tool = tool
        self.model_menu.value = self._segmentation_tool.model_menu.value
        self._surface_chosen()
        self._segmentation_tool.segmentation_cursors[self.axis].radius = self.segmentation_overlay.radius
        self._segmentation_tool.setCursorOffsetFromOrigin(
            self.axis, self.mvSegmentationCursorOffsetFromOrigin()
        )
        # TODO:
        # Set the segmentation pucks' locations based on the current slice location
        # self._segmentation_tool.segmentation_cursors[self.axis].
        self.segmentation_cursor_enabled = True
        self.view.redraw_needed = True
        # Set their radii to the current selected models' thickness
        # Synchronize the tool's model menu value to our model menu value

    def mvSegmentationCursorOffsetFromOrigin(self):
        return self.view.drawing.parent.data.dicom_data.origin()[self.axis] + (self.pos * self.view.drawing.parent.data.step[self.axis])

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
            self.segmentation_tool.setCursorOffsetFromOrigin(self.axis, self.mvSegmentationCursorOffsetFromOrigin()) #self.pos * self.view.drawing.parent.data.step[self.axis]
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
        if (
                ww <= 0 or wh <= 0 or width <= 0 or height <= 0
                # below: temporary workaround for #2162
                or self.view is None or self.view.render is None
        ):
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
            self.vertical_slice_overlay.top
            , self.vertical_slice_overlay.bottom
            , self.horizontal_slice_overlay.bottom
            , self.horizontal_slice_overlay.top
        )
        return top, bottom, left, right

    def event(self, event):
        if event.type() == QEvent.Type.Enter:
            self.enterEvent()
        if event.type() == QEvent.Type.Leave:
            self.leaveEvent()
        return QWindow.event(self, event)

    @property
    def guidelines_visible(self):
        return self.vertical_slice_overlay.display

    def setGuidelineVisibility(self, visibility: bool):
        self.horizontal_slice_overlay.display = visibility
        self.vertical_slice_overlay.display = visibility

    def enableSegmentationOverlay(self):
        self.segmentation_overlay.display = True

    def disableSegmentationOverlay(self):
        self.segmentation_overlay.display = False

    def enterEvent(self):
        if self.shouldEnableSegmentationCursor():
            self.enableSegmentationOverlay()

    def leaveEvent(self):
        self.disableSegmentationOverlay()

    # Context Menu Requirements:
    # -- Should open at the site of the original right click
    # -- Should not open if the mouse has been moved since the right click,
    #    as the user was probably zooming in and out
    # -- The leaveEvent from going into the context menu shouldn't hide the
    #    segmentation overlay; at least, if it's unavoidable triggering it,
    #    then it should turn back on after the context menu closes

    def shouldEnableSegmentationCursor(self):
        return self.segmentation_cursor_enabled

    def shouldOpenContextMenu(self):
        return (
            self.context_menu_coords is not None
            and not self.mouse_moved_during_right_click
        )

    def mousePressEvent(self, event):  # noqa
        b = event.button() | event.buttons()
        if b & Qt.MouseButton.RightButton:
            self.context_menu_coords = self.widget.mapToGlobal(event.pos())
        if b & Qt.MouseButton.MiddleButton:
            pass
        if b & Qt.MouseButton.LeftButton:
            # No idea why but if we don't hide it by ourselves here the event system won't take
            # care of it for us
            pass

    def mouseReleaseEvent(self, event): # noqa
        b = event.button() | event.buttons()
        if b & Qt.MouseButton.RightButton:
            #if self.shouldOpenContextMenu():
            #    from Qt.QtWidgets import QMenu, QAction
            #    if not self.context_menu:
            #        self.context_menu = QMenu(parent=self.parent)
            #        toggle_guidelines_action = QAction("Toggle Guidelines")
            #        self.context_menu.addAction(toggle_guidelines_action)
            #        toggle_guidelines_action.triggered.connect(lambda: self.manager.toggle_guidelines())
            #        self.context_menu.aboutToHide.connect(self.enterEvent)
            #    self.context_menu.exec(self.context_menu_coords)
            #    self.mouse_moved_during_right_click = False
            #self.mouse_moved_during_right_click = False
            pass
        if b & Qt.MouseButton.LeftButton:
            self.segmentation_overlay.center = (self.scale * event.position().x(), self.scale * (self.view.window_size[1] - event.position().y()), 0)
            self.segmentation_overlay.update()
            self.view.camera.redraw_needed = True
        self.last_mouse_position = None

    def resize3DSegmentationCursor(self):
        if self.segmentation_tool:
            ww, wh = self.main_view.window_size
            ps = self.main_view.pixel_size()
            width, height = self.view.window_size
            psize = self.view.pixel_size()
            radius = self.segmentation_overlay.absolute_radius
            rel_size = (radius / width) * psize
            needed_rad = (rel_size / psize) * ww
            self.segmentation_tool.segmentation_cursors[self.axis].radius = self.segmentation_overlay.radius * psize / ps # 0.5 * needed_rad

    def wheelEvent(self, event):
        # Looked like the same size with:
        # self.view.pixel_size = 0.14930555555555
        # self.segmentation_overlay.absolute_radius = 80
        # self.segmentation_tool.segmentation_cursor.radius = 5.9722222222
        #
        modifier = event.modifiers()
        delta = event.angleDelta()
        x_dir, y_dir = np.sign(delta.x()), np.sign(delta.y())
        if modifier == Qt.KeyboardModifier.ShiftModifier:
            self.segmentation_overlay.absolute_radius += 1 * x_dir
            self.segmentation_overlay.pixel_size = self.view.pixel_size()
            self.resize3DSegmentationCursor()
        elif modifier == Qt.KeyboardModifier.NoModifier:
            self.segmentation_overlay.pixel_size = self.view.pixel_size()
            self.view.camera.field_width += 1 * y_dir
            self.resize3DSegmentationCursor()
        self.view.camera.redraw_needed = True

    def mousePercentOffsetsFromEdges(self, x, y):
        top, bottom, left, right = self.camera_space_drawing_bounds()
        percent_offset_from_bottom = (((self.scale * (self.view.window_size[1] - y)) - bottom) / (top - bottom))
        percent_offset_from_top = 1 - percent_offset_from_bottom
        percent_offset_from_left = (((self.scale * x) - left) / (right - left))
        percent_offset_from_right = 1 - percent_offset_from_left
        return percent_offset_from_top, percent_offset_from_bottom, percent_offset_from_left, percent_offset_from_right

    def cameraSpaceDrawingOffsets(self):
        psize = self.view.pixel_size()
        if self.axis == Axis.SAGGITAL:
            # TODO: Why does this need a constant scale of 2 regardless of self.scale?
            x_offset = self.camera_offsets[Axis.CORONAL] * 2 / psize
            y_offset = self.camera_offsets[Axis.AXIAL] * 2 / psize
        elif self.axis == Axis.CORONAL:
            x_offset = self.camera_offsets[Axis.SAGGITAL] * 2 / psize
            y_offset = self.camera_offsets[Axis.AXIAL] * 2 / psize
        else:
            x_offset = self.camera_offsets[Axis.SAGGITAL] * 2 / psize
            y_offset = self.camera_offsets[Axis.CORONAL] * 2 / psize
        return x_offset, y_offset

    def mouseMoveEvent(self, event):  # noqa
        b = event.button() | event.buttons()
        # Level or segment
        if b == Qt.MouseButton.NoButton or b == Qt.MouseButton.LeftButton:
            pos = event.position()
            x, y = pos.x(), pos.y()
            self.segmentation_overlay.center = (self.scale * x, self.scale * (self.view.window_size[1] - y), 0)
            self.segmentation_overlay.update()
            if self.segmentation_tool:
                top, bottom, left, right = self.camera_space_drawing_bounds()
                rel_top, rel_bottom, rel_left, rel_right = self.mousePercentOffsetsFromEdges(x, y)
                x_offset, y_offset = self.cameraSpaceDrawingOffsets()
                # TODO Why did I have to add the y-offset here but not the x-offset?
                if left <= self.scale * x <= right and bottom <= (self.scale * (y + y_offset)) <= top:
                    old_origin = self.segmentation_tool.segmentation_cursors[self.axis].origin
                    drawing_origin = self.view.drawing.parent.data.dicom_data.origin()
                    origin = old_origin
                    if self.axis == Axis.AXIAL:
                        absolute_offset_left = rel_right * self.dimensions[0] * self.view.drawing.parent.data.step[0]
                        absolute_offset_bottom = rel_top * self.dimensions[1] * self.view.drawing.parent.data.step[1]
                        origin[0], origin[1] = absolute_offset_left + drawing_origin[0], absolute_offset_bottom + drawing_origin[1]
                    if self.axis == Axis.CORONAL:
                        absolute_offset_left = rel_left * self.dimensions[0] * self.view.drawing.parent.data.step[0]
                        absolute_offset_bottom = rel_bottom * self.dimensions[2] * self.view.drawing.parent.data.step[2]
                        origin[0], origin[2] = drawing_origin[0] + absolute_offset_left, drawing_origin[2] + absolute_offset_bottom
                    if self.axis == Axis.SAGGITAL:
                        absolute_offset_left = rel_left * self.dimensions[1] * self.view.drawing.parent.data.step[1]
                        absolute_offset_bottom = rel_bottom * self.dimensions[2] * self.view.drawing.parent.data.step[2]
                        origin[1], origin[2] = drawing_origin[1] + absolute_offset_left, drawing_origin[2] + absolute_offset_bottom
                    self.segmentation_tool.segmentation_cursors[self.axis].origin = origin
            self.view.camera.redraw_needed = True
        # Zoom / Dolly
        if b & Qt.MouseButton.RightButton:
            self.mouse_moved_during_right_click = True
            pos = event.position()
            x, y = pos.x(), pos.y()
            if not self.last_mouse_position:
                dy = 0
            else:
                dy = y - self.last_mouse_position[1]
            self.last_mouse_position = [x, y]
            self.view.camera.field_width += 1 * np.sign(dy) # offsets[self.axis] += (-dy * psize) * 3 * self.axis.positive_direction
            #self.resize3DSegmentationCursor()
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
