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

# TODO: Don't rely on the frame drawn trigger to redraw this view. Although
# more convenient, the constant passing around of the context results in the
# ChimeraX UI flickering.
# TODO: There are a lot of little hacks that would go away entirely if we
# could base the currently displayed orthoplane slice on the GridData directly
# instead of abusing the model system's Texture2DPlanes code to get orthoplanes
# generated.
# We would no longer have to coordinate hiding the orthoplanes model in the 3D view
#     then showing it in the 2D viewports.
# We would no longer have to move the camera in and out when the current slice changes
# We could display more than one model's orthoplane at a time, say if a physician wanted
#     to compare two different CT scans.
import numpy as np
from math import sqrt

from Qt import qt_object_is_deleted
from Qt.QtCore import Qt, QEvent, QSize
from Qt.QtGui import QContextMenuEvent, QWindow, QSurface
from Qt.QtWidgets import QVBoxLayout, QHBoxLayout, QWidget, QSlider, QLabel
from Qt.QtCore import QTimer, QPoint

from chimerax.core.commands import log_equivalent_command
from chimerax.core.models import Surface
from chimerax.geometry import Place, translation
from chimerax.graphics import Drawing
from chimerax.map import Volume, VolumeSurface, VolumeImage
from chimerax.map.volume import show_planes
from chimerax.map.volume_viewer import VolumeViewer, Histogram_Pane
from chimerax.map.volumecommand import apply_volume_options
from chimerax.ui.widgets import ModelMenu

from ..graphics import (
    OrthoplaneView, OrthoCamera
    , SegmentationOverlay, SegmentationCursorOverlay, SegmentationCursorOnOtherAxisOverlay
    , OrthoplaneLocationOverlay
)
from ..types import Direction, Axis
from .label import Label

class LevelLabel(QLabel):
    def __init__(self, graphics_window):
        super().__init__()
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.ToolTip)
        self.graphics_window = graphics_window

    def show_text(self, text, x, y):
        self.setText(text)
        self.move(self.graphics_window.mapToGlobal(QPoint(int(x)+10, int(y))))
        self.show()

class PlaneViewerManager:
    # TODO: If the drawings in the orthoplane viewers are not the same as the drawings in the
    # 3D window, this class actually needs to take care of managing which drawings are in which
    # viewer.
    def __init__(self, session):
        self.session = session
        self.have_seg_tool = False
        self.axes = {}
        self.volumes = {}

    def register(self, viewer):
        self.axes[viewer.axis] = viewer

    def update_location(self, viewer):
        if viewer.axis == Axis.AXIAL:
            if Axis.CORONAL in self.axes:
                self.axes[Axis.CORONAL].axial_index = viewer.axial_index
            if Axis.SAGITTAL in self.axes:
                self.axes[Axis.SAGITTAL].axial_index = viewer.axial_index
        if viewer.axis == Axis.CORONAL:
            if Axis.AXIAL in self.axes:
                self.axes[Axis.AXIAL].coronal_index = viewer.coronal_index
            if Axis.SAGITTAL in self.axes:
                self.axes[Axis.SAGITTAL].coronal_index = viewer.coronal_index
        if viewer.axis == Axis.SAGITTAL:
            if Axis.AXIAL in self.axes:
                self.axes[Axis.AXIAL].sagittal_index = viewer.sagittal_index
            if Axis.CORONAL in self.axes:
                self.axes[Axis.CORONAL].sagittal_index = viewer.sagittal_index

    def update_dimensions(self, dimensions):
        for axis in self.axes.values():
            axis.update_dimensions(dimensions)

    def register_segmentation_tool(self, tool):
        for viewer in self.axes.values():
            viewer.segmentation_tool = tool
        self.have_seg_tool = True

    def clear_segmentation_tool(self):
        for viewer in self.axes.values():
            viewer.segmentation_tool = None
        self.have_seg_tool = False

    def segmentation_tool(self):
        if not self.have_seg_tool:
            return None
        return self.axes[Axis.AXIAL].segmentation_tool

    def toggle_guidelines(self):
        layout = self.session.ui.main_window.main_view.view_layout()
        if self.axes[Axis.AXIAL].guidelines_visible:
            log_equivalent_command(self.session, f"dicom view {layout} guidelines false")
        else:
            log_equivalent_command(self.session, f"dicom view {layout} guidelines true")
        for viewer in self.axes.values():
            viewer.setGuidelineVisibility(not viewer.guidelines_visible)

    def show_guidelines(self):
        for viewer in self.axes.values():
            viewer.setGuidelineVisibility(True)

    def hide_guidelines(self):
        for viewer in self.axes.values():
            viewer.setGuidelineVisibility(False)

    def update_displayed_model(self, model):
        for viewer in self.axes.values():
            viewer.model_menu._menu.set_value(model)

    def add_segmentation(self, seg):
        for viewer in self.axes.values():
            viewer.add_segmentation(seg)

    def remove_segmentation(self, seg):
        for viewer in self.axes.values():
            viewer.remove_segmentation(seg)

    def update_segmentation_overlay_for_segmentation(self, segmentation):
        for viewer in self.axes.values():
            viewer.segmentation_overlays[segmentation].needs_update = True
            viewer.render()

    def redraw_all(self):
        for viewer in self.axes.values():
            viewer.render()

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
        self.level_label = LevelLabel(self.widget)
        self.setSurfaceType(QSurface.SurfaceType.OpenGLSurface)
        self.placeholder_drawing = Drawing("placeholder")
        self.view = OrthoplaneView(self.placeholder_drawing, window_size = (0, 0), axis = self.axis)
        self.view.initialize_rendering(session.main_view.render.opengl_context)
        self.view.camera = OrthoCamera()
        self.field_width_offset = 0

        camera = self.view.camera
        camera.position = Place(origin = (0,0,0), axes = self.axes)

        # TODO: Set to main_view background color and update it in render loop
        self.view.background_color = (0, 0, 0, 255)

        self.main_view = session.main_view

        self.camera_offsets = [0, 0, 0]
        self._plane_indices = [0, 0, 0]

        self.drawings = []
        self.current_segmentation_cursor_overlays = []

        self.label = Label(self.session, self.view, str(axis), str(axis), size=16, xpos=0, ypos=0)

        def _not_volume_surface_or_segmentation(m):
            ok_to_list = not isinstance(m, VolumeSurface)
            # This will run over all models which may not have DICOM data...
            try:
                if hasattr(m.data, "dicom_data"):
                    ok_to_list &= not m.data.dicom_data.dicom_series.modality == "SEG"
                    ok_to_list &= not m.data.reference_data
            except AttributeError:
                pass
            return ok_to_list

        self.model_menu = ModelMenu(
            self.session, parent, label = 'Model',
            model_types = [Volume, Surface],
            model_filter = _not_volume_surface_or_segmentation,
            model_chosen_cb = self._surfaceChosen
        )

        # TODO: Create these on demand when the segmentation tool gets registered
        self.segmentation_cursor_overlay = SegmentationCursorOverlay("seg_overlay", radius=20, thickness=3)
        self.horizontal_slice_overlay = OrthoplaneLocationOverlay("horiz_overlay", slice=10, direction=Direction.HORIZONTAL)
        self.vertical_slice_overlay = OrthoplaneLocationOverlay("vertical_overlay", slice=11)
        self.segmentation_cursor_other_axis_horizontal_overlay = SegmentationCursorOnOtherAxisOverlay("seg_cursor_other_axis_horizontal", direction=Direction.HORIZONTAL)
        self.segmentation_cursor_other_axis_vertical_overlay = SegmentationCursorOnOtherAxisOverlay("seg_cursor_other_axis_vertical", direction=Direction.VERTICAL)
        self.horizontal_slice_overlay.display = False
        self.vertical_slice_overlay.display = False
        self.segmentation_cursor_other_axis_horizontal_overlay.display = False
        self.segmentation_cursor_other_axis_vertical_overlay.display = False
        self.segmentation_cursor_overlay.display = False
        self.view.add_cursor_overlay(self.segmentation_cursor_overlay)
        self.view.add_guideline_overlay(self.horizontal_slice_overlay)
        self.view.add_guideline_overlay(self.vertical_slice_overlay)
        self.view.add_guideline_overlay(self.segmentation_cursor_other_axis_horizontal_overlay)
        self.view.add_guideline_overlay(self.segmentation_cursor_other_axis_vertical_overlay)
        self.segmentation_overlays: dict[Volume, SegmentationOverlay] = {}

        self.mouse_move_timer = QTimer()
        self.mouse_move_timer.setInterval(500);
        self.mouse_move_timer.setSingleShot(True);
        self.mouse_move_timer.timeout.connect(self.mouseStoppedMoving)
        self.mouse_x = 0
        self.mouse_y = 0
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
        if axis == Axis.SAGITTAL:
            self.slider.setMaximum(max_x)
            self.slider.setValue(1)

        self.slider.sliderMoved.connect(self._onSliderMoved)
        self.slider_moved = False

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
        #if self.view.drawing is placeholder_drawing:
        #    self._surfaceChosen()

    @property
    def segmentation_tool(self):
        return self._segmentation_tool

    @segmentation_tool.setter
    def segmentation_tool(self, tool):
        self._segmentation_tool = tool
        if tool is not None:
            if self.model_menu.value != self._segmentation_tool.model_menu.value or self.view.drawing is self.placeholder_drawing:
                self.model_menu.value = self._segmentation_tool.model_menu.value
            self._segmentation_tool.segmentation_cursors[self.axis].radius = self.segmentation_cursor_overlay.radius
            # TODO:
            # Set the segmentation pucks' locations based on the current slice location
            # self._segmentation_tool.segmentation_cursors[self.axis].
            self.segmentation_cursor_overlay.display = True
            self.view.redraw_needed = True
            # Set their radii to the current selected models' thickness
            # Synchronize the tool's model menu value to our model menu value

    def mvSegmentationCursorOffsetFromOrigin(self):
        origin = self.drawingOrigin()
        return self.drawingOrigin()[self.axis] + (self.pos * self.drawingVolumeStep()[self.axis])

    def _onSliderMoved(self):
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
            if not self.view.drawing.parent.data.inferior_to_superior:
                diff = -diff
        #self.camera_offsets[self.axis] -= diff * self.drawingVolumeStep()[self.axis]
        # TODO: Set the segmentation drawing's position to coincide with the new slice
        if self.segmentation_tool:
            self.segmentation_tool.setCursorOffsetFromOrigin(self.axis, self.mvSegmentationCursorOffsetFromOrigin()) #self.pos * self.view.drawing.parent.data.step[self.axis]
        self._plane_indices[self.axis] = self.pos
        if self.segmentation_tool:
            for segmentation in self.segmentation_overlays.values():
                segmentation.slice = self.pos
        self.manager.update_location(self)
        if self.guidelines_visible or self.segmentation_tool:
            self.manager.redraw_all()
        else:
            self.render()
        self._redraw()

    def close(self):
        # TODO: why does this call make it crash?
        # self.setParent(None)
        self.label.delete()
        del self.label
        volume_viewer = None
        for tool in self.session.tools:
            if type(tool) == VolumeViewer:
                volume_viewer = tool
                break
        v = self.view.drawing.parent
        if self.view.drawing is not self.placeholder_drawing:
            self._remove_axis_from_volume_viewer(volume_viewer, v)
        self.view.drawing.delete()
        self.view.delete()
        QWindow.destroy(self)

    def _redraw(self, *_):
        self.render()

    def update_and_rerender(self):
        self.view.drawing.parent.update_drawings()
        self.render()

    #region drawing info
    def drawingOrigin(self):
        return self.drawingParentVolume().data.origin

    def drawingParentVolume(self, drawing = None):
        if drawing:
            return drawing.parent
        return self.view.drawing.parent

    def drawingVolumeStep(self):
        return self.drawingParentVolume().data.step

    def drawingPosition(self):
        return self.view.drawing.position # _child_drawings[0]._child_drawings[0].position

    def drawingBounds(self):
        return self.view.drawing.bounds() # _child_drawings[0]._child_drawings[0].bounds()
    #endregion

    def exposeEvent(self, event):
        if self.isExposed() and not self.session.update_loop.blocked():
            self.render()

    def resizeEvent(self, event):  # noqa
        size = event.size()
        width = size.width()
        height = size.height()
        self.setViewport(width, height)

    def setViewport(self, width, height):
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
            self.scale = mvwin.opengl_context.pixel_scale()
            # TODO: If the user selects 'surface' then 'orthoplanes' in the volume viewer we should
            # override the default plane locations somehow
            if self.slider_moved:
                for d in self.drawings:
                    show_planes(self.drawingParentVolume(d), self.axis, self._plane_indices[self.axis])
                    self.drawingParentVolume(d).update_drawings()
                self.slider_moved = False
            model_center_offsets = self.drawingBounds().center()
            model_sizes = self.drawingBounds().size()
            initial_needed_fov = model_sizes[self.axis.vertical] / height * width
            margin = 24
            self.view.camera.field_width = initial_needed_fov + margin + self.field_width_offset
            self.calculateSliceOverlays()
            # The camera and the model can't share the same origin, so back it off a little bit
            camera_offsets = [0, 0, 0]
            camera_offsets[self.axis] = 20 * self.axis.positive_direction
            self.origin = self.drawingPosition().origin() + model_center_offsets - self.camera_offsets + camera_offsets
            camera = self.view.camera
            camera.position = Place(axes=self.axes, origin=self.origin)
            self.segmentation_cursor_overlay.update()
            self.horizontal_slice_overlay.update()
            self.vertical_slice_overlay.update()

            self.view.prepare_scene_for_drawing()
            self.view._draw_scene(self.view.camera, [self.view.drawing])
            self.view.finalize_draw()
        except Exception as e: # noqa
            # This line is here so you can set a breakpoint on it and figure out what's going wrong
            # because ChimeraX's interface will not tell you.
            pass
        finally:
            # Target opengl context back to main graphics window.
            self.main_view.render.use_shared_context(mvwin)
        self.view.render.done_current()

    def toggle_guidelines(self):
        if self.segmentation_tool:
            self.segmentation_tool.setGuidelineCheckboxValue(not self.guidelines_visible)
        else:
            self.manager.toggle_guidelines()

    def add_segmentation(self, segmentation):
        self.segmentation_overlays[segmentation] = SegmentationOverlay(segmentation.name + "_overlay", segmentation, self.axis)
        self.view.add_overlay(self.segmentation_overlays[segmentation])
        self.segmentation_overlays[segmentation].slice = self.pos

    def remove_segmentation(self, segmentation):
        self.view.remove_overlays([self.segmentation_overlays[segmentation]])
        del self.segmentation_overlays[segmentation]

    def addDrawing(self, drawing):
        self.drawings.append(drawing)

    def removeDrawing(self, drawing):
        self.drawings.remove(drawing)

    def calculateSliceOverlays(self):
        width, height = self.view.window_size
        size = self.view.drawing.bounds().size()
        psize = self.view.pixel_size()
        axis_sizes = (size / psize)[::-1]
        x_offset, y_offset = self.cameraSpaceDrawingOffsets()
        if self.axis == Axis.SAGITTAL:
            self.horizontal_slice_overlay.bottom = 0.5 * self.scale * ((width + x_offset) - axis_sizes[Axis.CORONAL])
            self.horizontal_slice_overlay.top =    0.5 * self.scale * ((width + x_offset) + axis_sizes[Axis.CORONAL])
            self.horizontal_slice_overlay.offset = 0.5 * self.scale * ((height + y_offset) - axis_sizes[Axis.SAGITTAL])
            self.horizontal_slice_overlay.tick_thickness = axis_sizes[Axis.SAGITTAL] / (self.dimensions[Axis.AXIAL] / self.scale)

            self.vertical_slice_overlay.bottom = 0.5 * self.scale * ((height + y_offset) - axis_sizes[Axis.SAGITTAL])
            self.vertical_slice_overlay.top =    0.5 * self.scale * ((height + y_offset) + axis_sizes[Axis.SAGITTAL])
            self.vertical_slice_overlay.offset = 0.5 * self.scale * ((width + x_offset) - axis_sizes[Axis.CORONAL])
            self.vertical_slice_overlay.tick_thickness = axis_sizes[Axis.CORONAL] / (self.dimensions[Axis.SAGITTAL] / self.scale)
        elif self.axis == Axis.CORONAL:
            self.horizontal_slice_overlay.bottom = 0.5 * self.scale * ((width + x_offset) - axis_sizes[Axis.AXIAL])
            self.horizontal_slice_overlay.top =    0.5 * self.scale * ((width + x_offset) + axis_sizes[Axis.AXIAL])
            self.horizontal_slice_overlay.offset = 0.5 * self.scale * ((height + y_offset) - axis_sizes[Axis.SAGITTAL])
            self.horizontal_slice_overlay.tick_thickness = axis_sizes[Axis.SAGITTAL] / (self.dimensions[Axis.AXIAL] / self.scale)

            self.vertical_slice_overlay.bottom = 0.5 * self.scale * ((height + y_offset) - axis_sizes[Axis.SAGITTAL])
            self.vertical_slice_overlay.top =    0.5 * self.scale * ((height + y_offset) + axis_sizes[Axis.SAGITTAL])
            self.vertical_slice_overlay.offset = 0.5 * self.scale * ((width + x_offset) - axis_sizes[Axis.AXIAL])
            self.vertical_slice_overlay.tick_thickness = axis_sizes[Axis.AXIAL] / (self.dimensions[Axis.SAGITTAL] / self.scale)
        else:
            self.horizontal_slice_overlay.bottom = 0.5 * self.scale * ((width - x_offset) - axis_sizes[Axis.AXIAL])
            self.horizontal_slice_overlay.top =    0.5 * self.scale * ((width - x_offset) + axis_sizes[Axis.AXIAL])
            self.horizontal_slice_overlay.offset = 0.5 * self.scale * ((height - y_offset) + axis_sizes[Axis.AXIAL])
            self.horizontal_slice_overlay.tick_thickness = axis_sizes[Axis.AXIAL] / (self.dimensions[Axis.CORONAL] / self.scale)

            self.vertical_slice_overlay.bottom = 0.5 * self.scale * ((height - y_offset) - axis_sizes[Axis.CORONAL])
            self.vertical_slice_overlay.top =    0.5 * self.scale * ((height - y_offset) + axis_sizes[Axis.CORONAL])
            self.vertical_slice_overlay.offset = 0.5 * self.scale * ((width - x_offset) + axis_sizes[Axis.CORONAL])
            self.vertical_slice_overlay.tick_thickness = axis_sizes[Axis.CORONAL] / (self.dimensions[Axis.SAGITTAL] / self.scale)
        # Update the segmentation overlays
        for overlay in self.segmentation_overlays.values():
            overlay.y_min = self.vertical_slice_overlay.bottom
            overlay.y_max = self.vertical_slice_overlay.top
            overlay.x_min = self.horizontal_slice_overlay.bottom
            overlay.x_max = self.horizontal_slice_overlay.top

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

    def enableSegmentationOverlays(self):
        self.segmentation_cursor_overlay.display = True

    def disableSegmentationOverlays(self):
        self.segmentation_cursor_overlay.display = False

    def enterEvent(self):
        if self.segmentation_tool:
            self.enableSegmentationOverlays()
            self.resize3DSegmentationCursor()
            self.segmentation_tool.make_puck_visible(self.axis)

    def leaveEvent(self):
        if self.segmentation_tool:
            self.disableSegmentationOverlays()
            self.segmentation_tool.make_puck_invisible(self.axis)
        self.mouse_move_timer.stop()

    def shouldOpenContextMenu(self):
        return (
            self.context_menu_coords is not None
            and not self.mouse_moved_during_right_click
        )

    def mousePressEvent(self, event):  # noqa
        b = event.button() | event.buttons()
        if self.context_menu:
            if not qt_object_is_deleted(self.context_menu):
                self.context_menu.close()
                self.context_menu = None
        if b & Qt.MouseButton.RightButton:
            self.context_menu_coords = self.widget.mapToGlobal(event.pos())
            if self.segmentation_tool and self.segmentation_cursor_overlay.display:
                self.segmentation_cursor_overlay.display = False
        if b & Qt.MouseButton.MiddleButton:
            if self.segmentation_tool and self.segmentation_cursor_overlay.display:
                self.segmentation_cursor_overlay.display = False
        if b & Qt.MouseButton.LeftButton:
            if self.segmentation_tool:
                x, y = event.position().x(), event.position().y()
                self.moveSegmentationPuck(x, y, record_seg = True)
        self._redraw()

    def mouseReleaseEvent(self, event): # noqa
        b = event.button() | event.buttons()
        modifier = event.modifiers()
        self.segmentation_cursor_overlay.center = (self.scale * event.position().x(), self.scale * (self.view.window_size[1] - event.position().y()), 0)
        self.segmentation_cursor_overlay.update()
        if b & Qt.MouseButton.RightButton:
            if self.shouldOpenContextMenu():
                from Qt.QtWidgets import QMenu, QAction
                if not self.context_menu:
                    self.context_menu = QMenu(parent=self.parent)
                    toggle_guidelines_action = QAction("Toggle Guidelines")
                    self.context_menu.addAction(toggle_guidelines_action)
                    toggle_guidelines_action.triggered.connect(lambda: self.toggle_guidelines())
                    self.context_menu.aboutToHide.connect(self.enterEvent)
                self.context_menu.exec(self.context_menu_coords)
                self.mouse_moved_during_right_click = False
            self.mouse_moved_during_right_click = False
        if b & Qt.MouseButton.LeftButton:
            if self.segmentation_tool:
                if modifier == Qt.KeyboardModifier.ShiftModifier:
                    self.segmentation_tool.removeMarkersFromSegment(self.axis, self.pos, self.current_segmentation_cursor_overlays)
                else:
                    self.segmentation_tool.addMarkersToSegment(self.axis, self.pos, self.current_segmentation_cursor_overlays)
                self.view.remove_cursor_overlays(self.current_segmentation_cursor_overlays)
                self.current_segmentation_cursor_overlays = []
                active_seg = self.segmentation_tool.active_seg
                self.manager.update_segmentation_overlay_for_segmentation(active_seg)
            self.view.camera.redraw_needed = True
        self.last_mouse_position = None
        if self.segmentation_tool:
            self.segmentation_cursor_overlay.display = True
        self._redraw()

    def resize3DSegmentationCursor(self):
        """Resize the 3D segmentation cursor based on the size of the 2D segmentation overlay."""
        # Does not depend on the pixel size in the main window
        if self.segmentation_tool:
            ww, wh = self.main_view.window_size
            width, height = self.view.window_size
            psize = self.view.pixel_size()
            radius = self.segmentation_cursor_overlay.radius
            rel_size = (radius / width) * psize
            needed_rad = (rel_size / psize) * ww
            self.segmentation_tool.segmentation_cursors[self.axis].radius = self.segmentation_cursor_overlay.radius * psize / self.scale

    def wheelEvent(self, event):
        modifier = event.modifiers()
        delta = event.angleDelta()
        x_dir, y_dir = np.sign(delta.x()), np.sign(delta.y())
        if modifier == Qt.KeyboardModifier.ShiftModifier:
            self.segmentation_cursor_overlay.radius += 1 * (x_dir | y_dir)
            self.resize3DSegmentationCursor()
        elif modifier == Qt.KeyboardModifier.NoModifier:
            self.field_width_offset += 1 * y_dir
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
        if self.axis == Axis.SAGITTAL:
            # TODO: Why does this need a constant scale of 2 regardless of self.scale?
            x_offset = self.camera_offsets[Axis.CORONAL] * 2 / psize
            y_offset = self.camera_offsets[Axis.AXIAL] * 2 / psize
        elif self.axis == Axis.CORONAL:
            x_offset = self.camera_offsets[Axis.SAGITTAL] * 2 / psize
            y_offset = self.camera_offsets[Axis.AXIAL] * 2 / psize
        else:
            x_offset = self.camera_offsets[Axis.SAGITTAL] * 2 / psize
            y_offset = self.camera_offsets[Axis.CORONAL] * 2 / psize
        return x_offset, y_offset

    def recordSegment(self, x, y):
        thisSegment = SegmentationCursorOverlay(
            "seg_overlay_" + str(len(self.current_segmentation_cursor_overlays)),
            radius=self.segmentation_cursor_overlay.radius, thickness=3
        )
        thisSegment.drawing_center = [x, y]
        thisSegment.center = self.segmentation_cursor_overlay.center
        self.current_segmentation_cursor_overlays.append(thisSegment)
        thisSegment.update()
        self.view.add_cursor_overlay(thisSegment)

    def moveSegmentationPuck(self, x, y, record_seg):
        top, bottom, left, right = self.camera_space_drawing_bounds()
        rel_top, rel_bottom, rel_left, rel_right = self.mousePercentOffsetsFromEdges(x, y)
        x_offset, y_offset = self.cameraSpaceDrawingOffsets()
        # TODO Why did I have to add the y-offset here but not the x-offset?
        if self.axis == Axis.AXIAL:
            y_offset = -y_offset
        if left <= self.scale * x <= right and bottom <= (self.scale * (y + y_offset)) <= top:
            old_origin = self.segmentation_tool.segmentation_cursors[self.axis].origin
            drawing_origin = self.drawingOrigin()
            origin = old_origin
            volume_steps = self.drawingVolumeStep()
            if self.axis == Axis.AXIAL:
                absolute_offset_left = rel_right * self.dimensions[0]
                absolute_offset_bottom = rel_top * self.dimensions[1]
                origin[0] = drawing_origin[0] + absolute_offset_left * volume_steps[0]
                origin[1] = drawing_origin[1] + absolute_offset_bottom * volume_steps[1]
            elif self.axis == Axis.CORONAL:
                absolute_offset_left = rel_left * self.dimensions[0]
                absolute_offset_bottom = rel_bottom * self.dimensions[2]
                origin[0] = drawing_origin[0] + absolute_offset_left * volume_steps[0]
                origin[2] = drawing_origin[2] + absolute_offset_bottom * volume_steps[2]
            else:  # self.axis == Axis.SAGITTAL:
                absolute_offset_left = rel_left * self.dimensions[1]
                absolute_offset_bottom = rel_bottom * self.dimensions[2]
                origin[1] = drawing_origin[1] + absolute_offset_left * volume_steps[0]
                origin[2] = drawing_origin[2] + absolute_offset_bottom * volume_steps[2]
            self.segmentation_tool.segmentation_cursors[self.axis].origin = origin
            if record_seg:
                self.recordSegment(absolute_offset_left, absolute_offset_bottom)

    def mouseStoppedMoving(self):
        # TODO: Look at the pixel under the mouse and report what the level is
        rel_top, rel_bottom, rel_left, rel_right = self.mousePercentOffsetsFromEdges(self.mouse_x, self.mouse_y)
        if any([rel_top < 0, rel_bottom < 0, rel_left < 0, rel_right < 0]) or any([rel_top > 1, rel_bottom > 1, rel_left > 1, rel_right > 1]):
            return
        if self.axis == Axis.AXIAL:
            absolute_offset_left = rel_right * self.dimensions[0]
            absolute_offset_bottom = rel_top * self.dimensions[1]
            level = self.drawingParentVolume().data.matrix()[self.pos, int(absolute_offset_bottom), int(absolute_offset_left)]
        elif self.axis == Axis.CORONAL:
            absolute_offset_left = rel_left * self.dimensions[0]
            absolute_offset_bottom = rel_bottom * self.dimensions[2]
            level = self.drawingParentVolume().data.matrix()[int(absolute_offset_bottom), self.pos, int(absolute_offset_left)]
        else:  # self.axis == Axis.SAGITTAL:
            absolute_offset_left = rel_left * self.dimensions[1]
            absolute_offset_bottom = rel_bottom * self.dimensions[2]
            level = self.drawingParentVolume().data.matrix()[int(absolute_offset_bottom), int(absolute_offset_left), self.pos]
        self.level_label.show_text("Level: " + str(level), self.mouse_x, self.mouse_y)

    def mouseMoveEvent(self, event):  # noqa
        self.level_label.hide()
        b = event.button() | event.buttons()
        # Level or segment
        pos = event.position()
        x, y = pos.x(), pos.y()
        self.mouse_x = x
        self.mouse_y = y
        if b == Qt.MouseButton.NoButton or b == Qt.MouseButton.LeftButton:
            self.segmentation_cursor_overlay.center = (self.scale * x, self.scale * (self.view.window_size[1] - y), 0)
            self.segmentation_cursor_overlay.update()
            if self.segmentation_tool:
                self.moveSegmentationPuck(x, y, record_seg = (b == Qt.MouseButton.LeftButton))
            self.view.camera.redraw_needed = True
            self.mouse_move_timer.start();

        # Zoom / Dolly
        if b & Qt.MouseButton.RightButton:
            self.mouse_moved_during_right_click = True
            if not self.last_mouse_position:
                dy = 0
            else:
                dy = y - self.last_mouse_position[1]
            self.last_mouse_position = [x, y]
            self.field_width_offset += 1 * np.sign(dy) # offsets[self.axis] += (-dy * psize) * 3 * self.axis.positive_direction
            #self.resize3DSegmentationCursor()
            self.view.camera.redraw_needed = True
        # Truck & Pedestal
        if b & Qt.MouseButton.MiddleButton:
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
            if self.axis == Axis.SAGITTAL:
                self.camera_offsets = [x, y + dx, z - dy]
        self._redraw()

    def keyPressEvent(self, event):  # noqa
        key = event.key()
        modifier = event.modifiers
        diff = 1
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            diff = 10
        if key == Qt.Key.Key_Right or key == Qt.Key.Key_D:
            event.accept()
            self.slider.setSliderDown(True)
            self.slider.setSliderPosition(self.pos * self.axis.positive_direction + diff)
            self.slider.setSliderDown(False)
            return
        if key == Qt.Key.Key_Left or key == Qt.Key.Key_A:
            event.accept()
            self.slider.setSliderDown(True)
            self.slider.setSliderPosition(self.pos * self.axis.positive_direction - diff)
            self.slider.setSliderDown(False)
            return
        else:
            return self.session.ui.forward_keystroke(event)

    def update_dimensions(self, dimensions):
        self.dimensions = dimensions

    #region index getters and setters
    @property
    def axial_index(self):
        return self._plane_indices[Axis.AXIAL]

    @property
    def coronal_index(self):
        return self._plane_indices[Axis.CORONAL]

    @property
    def sagittal_index(self):
        return self._plane_indices[Axis.SAGITTAL]

    @axial_index.setter
    def axial_index(self, index):
        self._plane_indices[Axis.AXIAL] = index
        # TODO: Have the overlay get this itself?
        if self.axis == Axis.CORONAL:
            self.horizontal_slice_overlay.slice = index
        if self.axis == Axis.SAGITTAL:
            self.horizontal_slice_overlay.slice = index

    @coronal_index.setter
    def coronal_index(self, index):
        self._plane_indices[Axis.CORONAL] = index
        # TODO: Have the overlay get this itself?
        if self.axis == Axis.AXIAL:
            self.horizontal_slice_overlay.slice = -index
        if self.axis == Axis.SAGITTAL:
            self.vertical_slice_overlay.slice = index

    @sagittal_index.setter
    def sagittal_index(self, index):
        self._plane_indices[Axis.SAGITTAL] = index
        # TODO: Have the overlay get this itself?
        if self.axis == Axis.AXIAL:
            self.vertical_slice_overlay.slice = -index
        if self.axis == Axis.CORONAL:
            self.vertical_slice_overlay.slice = index
    #endregion

    def _surfaceChosen(self, *args):
        # TODO: Create a copy of the parent study just for rendering in the orthoplane windows?
        # Would need to create a copy of each segmentation created, too, one for the orthoplane
        # windows and one for the
        # then modify the segmentation tool to
        volume_viewer = None
        for tool in self.session.tools:
            if type(tool) == VolumeViewer:
                volume_viewer = tool
        v = self.view.drawing
        if self.view.drawing is not self.placeholder_drawing:
            self._remove_axis_from_volume_viewer(volume_viewer, v)

        v = self.model_menu.value.copy()
        self.model_menu.value.expand_single_plane()
        self.model_menu.value.set_display_style('surface')
        self.model_menu.value._drawings_need_update()
        max_x, max_y, max_z = self.model_menu.value.region[1]
        middle = tuple((imin + imax) // 2 for imin, imax in zip(self.model_menu.value.region[0], self.model_menu.value.region[1]))
        new_drawing = None
        apply_volume_options(
            v
            , doptions = {
                'region': (v.region[0], v.region[1])
                , 'planes': self.axis.cartesian
            }
            , roptions = {}
            , image_mode_off = False
            , session = self.session
        )
        v.set_display_style('image')
        v.name =  str(self.axis) + " orthoplane " + str(self.model_menu.value.name)
        v.update_drawings()
        v.allow_style_changes = False
        # Add our new volume to the volume menu with our custom widget
        self._add_axis_to_volume_viewer(volume_viewer, v)

        self.main_view.camera.redraw_needed = True
        for d in v._child_drawings:
            if type(d) == VolumeImage:
                new_drawing = d
        #self.manager.update_drawing(self.model_menu.value)
        if new_drawing is not None:
            # Set the view's root drawing, and our ground truth drawing, to the new one
            self.view.drawing = new_drawing
            for drawing in self.drawings:
                self.removeDrawing(drawing)
            if new_drawing not in self.drawings:
                self.addDrawing(new_drawing)
                new_drawing.display = True
            self.set_label_text(new_drawing.parent.name)
            self.manager.update_dimensions([max_x, max_y, max_z])
            orthoplane_positions = max_x // 2, max_y // 2, max_z // 2
            self._plane_indices = list(orthoplane_positions)
            if self.axis == Axis.AXIAL:
                self.slider.setMaximum(max_z)
            if self.axis == Axis.CORONAL:
                self.slider.setRange(-max_y, 0)
            if self.axis == Axis.SAGITTAL:
                self.slider.setMaximum(max_x)
            self.slider.setValue(orthoplane_positions[self.axis] * self.axis.positive_direction)
            self.slider_moved = True
            self.pos = orthoplane_positions[self.axis]
            self._plane_indices[self.axis] = self.pos
            self.manager.update_location(self)
        self.render()

    def set_label_text(self, text):
        self.label.text = text
        self.label.update_drawing()

    def _remove_axis_from_volume_viewer(self, volume_viewer, volume):
        hptable = volume_viewer.thresholds_panel.histogram_table
        for v in tuple([volume]):
            if v in hptable:
                volume_viewer.thresholds_panel.close_histogram_pane(hptable[v])

    def _add_axis_to_volume_viewer(self, volume_viewer, volume):
        v = volume
        tp = volume_viewer.thresholds_panel
        hptable = tp.histogram_table
        if v in hptable:
          return

        if hasattr(v, 'series'):
            same_series = [vp for vp in hptable.keys()
                           if vp is not None and hasattr(vp, 'series') and vp.series == v.series]
        else:
            same_series = []

        if same_series:
          # Replace entry with same id number, for volume series
          vs = same_series[0]
          hp = hptable[vs]
          del hptable[vs]
        elif None in hptable:
          hp = hptable[None]                # Reuse unused histogram
          del hptable[None]
        elif len(hptable) >= tp.maximum_histograms():
          hp = tp.active_order[-1]        # Reuse least recently active histogram
          del hptable[hp.volume]
        else:
          # Make new histogram
          hp = SegmentationVolumePanel(self, tp.dialog, tp.histograms_frame, tp.histogram_height)
          hl = tp.histograms_layout
          hl.insertWidget(hl.count()-1, hp.frame)
          tp.histogram_panes.append(hp)

        hp.set_data_region(v)
        hptable[v] = hp
        tp.set_active_histogram(hp)
        tp._allow_panel_height_increase = True



class SegmentationVolumePanel(Histogram_Pane):
    """When a volume is added to a session it typically spawns the Volume Viewer, which
    gets populated with a panel allowing the user to control how the volume is rendered,
    which plane is shown if it's a plane, etc. This class is a wrapper around that panel
    that disables certain features."""
    def __init__(self, plane_viewer, dialog, parent, histogram_height):
        self.plane_viewer = plane_viewer
        self.dialog = dialog
        self.volume = None
        self.histogram_data = None
        self.histogram_size = None
        self._surface_levels_changed = False
        self._image_levels_changed = False
        self._log_moved_marker = False
        self.update_timer = None

        from Qt.QtWidgets import QVBoxLayout, QHBoxLayout, QFrame, QLabel, QPushButton, QMenu, QLineEdit, QSizePolicy
        from Qt.QtCore import Qt, QSize

        self.frame = f = QFrame(parent)
        f.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self._layout = flayout = QVBoxLayout(f)
        flayout.setContentsMargins(0,0,0,0)
        flayout.setSpacing(0)

        # Put volume name on separate line.
        self.data_name = nm = QLabel(f)
        flayout.addWidget(nm)
        nm.mousePressEvent = self.select_data_cb

        # Create frame for step, color, level controls.
        df = QFrame(f)
        df.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
#        df.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Minimum)
        flayout.addWidget(df)
        layout = QHBoxLayout(df)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(10)

        # Display / hide map button
        self.shown = sh = QPushButton(df)
        sh.setAttribute(Qt.WA_LayoutUsesWidgetRect) # Avoid extra padding on Mac
        sh.setMaximumSize(20,20)
        sh.setCheckable(True)
        sh.setFlat(True)
        sh.setStyleSheet('QPushButton {background-color: transparent;}')
        from chimerax.ui.icons import get_qt_icon
        sh_icon = get_qt_icon('shown')
        sh.setIcon(sh_icon)
        sh.setIconSize(QSize(20,20))
        sh.clicked.connect(self.show_cb)
        layout.addWidget(sh)
        sh.setToolTip('Display or undisplay data')
        self.shown_handlers = []

        # Color button
        from chimerax.ui.widgets import ColorButton
        cl = ColorButton(df, max_size = (16,16), has_alpha_channel = True, pause_delay = 1.0)
        self._color_button = cl
        cl.color_changed.connect(self._color_chosen)
        cl.color_pause.connect(self._log_color_command)
        layout.addWidget(cl)

        self.data_id = did = QLabel(df)
        layout.addWidget(did)
        did.mousePressEvent = self.select_data_cb

        self.size = sz = QLabel(df)
        layout.addWidget(sz)
        sz.mousePressEvent = self.select_data_cb

        # Subsampling step menu
        sl = QLabel('step', df)
        layout.addWidget(sl)
        import sys
        if sys.platform == 'darwin':
            # Setting padding cam make layout more compact on macOS 10.15 but makes clicking
            # on menu button down arrow do nothing.  So use default style on Mac.
            menu_button_style = None
        else:
            # Reduce button width and height on Windows and Linux
            menu_button_style = 'padding-left: 6px; padding-right: 6px; padding-top: 3px; padding-bottom: 3px'
        layout.addSpacing(-8)	# Put step menu closer to step label.
        self.data_step = dsm = QPushButton(df)
        if menu_button_style:
            dsm.setStyleSheet(menu_button_style)
        sm = QMenu(df)
        for step in (1,2,4,8,16):
            sm.addAction('%d' % step, lambda s=step: self.data_step_cb(s))
        dsm.setMenu(sm)
        layout.addWidget(dsm)

        # Threshold level entry
        lh = QLabel('Level', df)
        layout.addWidget(lh)
        layout.addSpacing(-5)	# Reduce padding to following entry field

        self.threshold = le = QLineEdit('', df)
        le.setMaximumWidth(40)
        le.returnPressed.connect(self.threshold_entry_enter_cb)
        layout.addWidget(le)

        self.data_range = rn = QLabel('? - ?', df)
        layout.addWidget(rn)

        # Display style menu
        self.style = stm = QPushButton(df)
        if menu_button_style:
            stm.setStyleSheet(menu_button_style)
        sm = QMenu(df)
        for style in ('surface', 'mesh', 'volume', 'maximum', 'plane', 'orthoplanes', 'box', 'tilted slab'):
            sm.addAction(style, lambda s=style: self.display_style_changed_cb(s))
        stm.setMenu(sm)
        layout.addWidget(stm)
        stm.setEnabled(False)
        stm.setVisible(False)

        layout.addStretch(1)

        # Close map button
        cb = QPushButton(df)
        cb.setAttribute(Qt.WA_LayoutUsesWidgetRect) # Avoid extra padding on Mac
        cb.setMaximumSize(20,20)
        cb.setFlat(True)
        layout.addWidget(cb)
        from chimerax.ui.icons import get_qt_icon
        cb_icon = get_qt_icon('minus')
        cb.setIcon(cb_icon)
        cb.setIconSize(QSize(20,20))
        cb.clicked.connect(self.close_map_cb)
        cb.setToolTip('Close data set')

        # Add histogram below row of controls
        h = self.make_histogram(f, histogram_height, new_marker_color = (1,1,1,1))
        flayout.addWidget(h)
        h.contextMenuEvent = self.show_context_menu

        # Create planes slider below histogram if requested.
        self._planes_slider_shown = False
        self._planes_slider_frame = None

    def show_plane_slider(self, show, axis = 2):
        pass

    def data_step_cb(self, step):
        self.data_step.setText('%d' % step)

        v = self.volume
        if v is None or v.region is None:
          return

        ijk_step = [step]
        if len(ijk_step) == 1:
          ijk_step = ijk_step * 3

        if tuple(ijk_step) == tuple(v.region[2]):
          return

        v.new_region(ijk_step = ijk_step, adjust_step = False)
        for vc in v.other_channels():
            vc.new_region(ijk_step = ijk_step, adjust_step = False)

        d = self.dialog
        if v != d.active_volume:
          d.display_volume_info(v)
        self.plane_viewer.update_and_rerender()

    def moved_marker_cb(self, marker):
        self.select_data_cb()	# Causes redisplay using GUI settings
        self.set_threshold_and_color_widgets()
        # Redraw graphics before more mouse drag events occur.
        self.plane_viewer.update_and_rerender()
