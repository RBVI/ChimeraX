# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2024 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This particular file is part of the ChimeraX library. You can also
# redistribute and/or modify it under the terms of the GNU Lesser General
# Public License version 2.1 as published by the Free Software Foundation.
# For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
# EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
# LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
# VERSION 2.1
#
# This notice must be embedded in or attached to all copies, including partial
# copies, of the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

"""
Lighting GUI tool for ChimeraX.

Provides a graphical interface for adjusting lighting and material properties,
similar to Chimera's Viewing Tool lighting controls.
"""

import sys
from enum import StrEnum

import numpy as np
from Qt.QtCore import Qt
from Qt.QtGui import QWindow, QSurface
from Qt.QtWidgets import (
    QHBoxLayout,
    QWidget,
    QLabel,
    QComboBox,
    QSlider,
    QFormLayout,
    QCheckBox,
    QTabWidget,
    QSizePolicy,
)

from chimerax.core.tools import ToolInstance
from chimerax.core.commands import run
from chimerax.ui import MainToolWindow
from chimerax.ui.widgets import ColorButton
from chimerax.geometry import Place
from chimerax.graphics import View, Drawing


# TODO: Move to graphics?
class LightingMode(StrEnum):
    Simple = "simple"
    Soft = "soft"
    Full = "full"
    Flat = "flat"
    Gentle = "gentle"


class SphereDrawing(Drawing):
    """A sphere for previewing lighting."""

    def __init__(self, facets: int = 8192):
        super().__init__("lighting_preview_sphere")
        from chimerax.surface import sphere_geometry2

        self.set_geometry(*sphere_geometry2(facets))
        self.color = (180, 180, 180, 255)  # Gray sphere


class LightArrowDrawing(Drawing):
    """An arrow showing light direction, positioned on sphere surface."""

    def __init__(self, name, color, sphere_radius=1.0):
        super().__init__(name)
        self.sphere_radius = sphere_radius
        self.arrow_length = 0.4
        self.arrow_radius = 0.04
        self.cone_radius = 0.08
        self.cone_height = 0.12
        self._direction = np.array([0, 0, -1], dtype=np.float32)
        self._create_arrow_geometry()
        self.color = color

    def _create_arrow_geometry(self):
        """Create arrow geometry (cylinder + cone)."""
        from chimerax.surface import cylinder_geometry, cone_geometry

        # Cylinder for shaft
        cyl_v, cyl_n, cyl_t = cylinder_geometry(
            radius=self.arrow_radius, height=self.arrow_length, nc=12, caps=True
        )
        # Shift cylinder so it starts at origin
        cyl_v[:, 2] += self.arrow_length / 2

        # Cone for head
        cone_v, cone_n, cone_t = cone_geometry(
            radius=self.cone_radius,
            height=self.cone_height,
            nc=12,
            caps=True,
            points_up=True,
        )
        # Position cone at end of cylinder
        cone_v[:, 2] += self.arrow_length + self.cone_height / 2

        # Combine geometries
        cone_t = cone_t + len(cyl_v)
        vertices = np.vstack([cyl_v, cone_v]).astype(np.float32)
        normals = np.vstack([cyl_n, cone_n]).astype(np.float32)
        triangles = np.vstack([cyl_t, cone_t]).astype(np.int32)

        self.set_geometry(vertices, normals, triangles)

        self.use_lighting = False

    @property
    def direction(self):
        return self._direction

    @direction.setter
    def direction(self, d):
        """Set arrow direction and update position on sphere."""
        d = np.asarray(d, dtype=np.float32)
        norm = np.linalg.norm(d)
        if norm > 0:
            d = d / norm
        self._direction = d
        self._update_position()

    def _update_position(self):
        """Position arrow on sphere surface pointing inward toward center."""
        d = self._direction
        # Light shines FROM direction d, so arrow should be at position
        # opposite to d (at -d * radius) and point TOWARD center (in direction +d)
        start_pos = -d * (self.sphere_radius + self.arrow_length + self.cone_height)

        # Create rotation to align arrow (which points +Z) with +d (toward center)
        target = d
        z_axis = np.array([0, 0, 1], dtype=np.float32)

        # Handle special case when target is close to z-axis
        if abs(np.dot(target, z_axis)) > 0.999:
            if target[2] > 0:
                rotation = Place()
            else:
                rotation = Place(axes=[[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        else:
            # Rotation axis is cross product
            axis = np.cross(z_axis, target)
            axis = axis / np.linalg.norm(axis)
            # Rotation angle
            angle = np.arccos(np.clip(np.dot(z_axis, target), -1, 1))
            from chimerax.geometry import rotation as make_rotation

            rotation = make_rotation(axis, np.degrees(angle))

        self.position = Place(origin=start_pos) * rotation


class LabeledSlider(QWidget):
    """A slider with a label showing its current value."""

    def __init__(self, min_val, max_val, default_val, decimals=2, parent=None):
        super().__init__(parent)
        self.decimals = decimals
        self.scale = 10**decimals

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(int(min_val * self.scale), int(max_val * self.scale))
        self.slider.setValue(int(default_val * self.scale))
        self.slider.valueChanged.connect(self._on_value_changed)

        self.value_label = QLabel(f"{default_val:.{decimals}f}")
        self.value_label.setMinimumWidth(40)

        layout.addWidget(self.slider, 1)
        layout.addWidget(self.value_label)

    def _on_value_changed(self, value):
        real_value = value / self.scale
        self.value_label.setText(f"{real_value:.{self.decimals}f}")

    def value(self):
        return self.slider.value() / self.scale

    def setValue(self, val):
        self.slider.setValue(int(val * self.scale))

    def valueChanged(self):
        """Return the valueChanged signal from the internal slider."""
        return self.slider.valueChanged

    def sliderReleased(self):
        """Return the sliderReleased signal from the internal slider."""
        return self.slider.sliderReleased


class LightingGUI(ToolInstance):
    """
    Lighting and Material GUI Tool.

    Provides controls for:
    - Lighting presets
    - Key/fill/ambient light intensities and colors
    - Material shininess (specular exponent and reflectivity)
    - Interactive lighting preview sphere
    """

    help = "help:user/tools/lighting.html"

    def __init__(self, session, tool_name="Lighting"):
        super().__init__(session, tool_name)
        self.tool_window = MainToolWindow(self)
        parent = self.tool_window.ui_area

        # Main layout: controls on left, preview on right
        main_layout = QHBoxLayout(parent)

        # Left side: tabbed controls
        self.tab_widget = QTabWidget()
        self.tab_widget.setMaximumWidth(280)
        # self.tab_widget.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)

        # Create tab contents
        lighting_tab = self._create_lighting_tab()
        shininess_tab = self._create_shininess_tab()
        depth_cue_tab = self._create_depth_cue_tab()

        self.tab_widget.addTab(lighting_tab, "Lighting")
        self.tab_widget.addTab(shininess_tab, "Shininess")
        self.tab_widget.addTab(depth_cue_tab, "Depth Cue")

        # Right side: preview sphere with light direction arrows
        self.preview_widget = LightingPreviewWidget(parent, session, self)

        main_layout.addWidget(self.tab_widget, 0)  # Fixed width controls
        main_layout.addWidget(self.preview_widget.widget, 1)  # Expandable preview

        # Sync UI with current state
        self._sync_from_session()

        self.tool_window.manage()

    def _create_lighting_tab(self):
        """Create the Lighting controls tab."""
        tab = QWidget()
        tab.setAutoFillBackground(True)
        layout = QFormLayout(tab)

        # Preset dropdown
        self.preset_combo = QComboBox()
        self.preset_combo.addItems([mode.value for mode in LightingMode])
        self.preset_combo.currentTextChanged.connect(self._on_preset_changed)
        layout.addRow("Preset:", self.preset_combo)

        # Key light intensity
        self.key_intensity = LabeledSlider(0, 1.5, 1.0, decimals=2)
        self.key_intensity.valueChanged().connect(self._on_key_intensity_changed)
        self.key_intensity.sliderReleased().connect(self._on_key_intensity_released)
        layout.addRow("Key intensity:", self.key_intensity)

        # Key light color
        self.key_color = ColorButton()
        self.key_color.color_changed.connect(self._on_key_color_changed)
        layout.addRow("Key color:", self.key_color)

        # Fill light intensity
        self.fill_intensity = LabeledSlider(0, 1.5, 0.5, decimals=2)
        self.fill_intensity.valueChanged().connect(self._on_fill_intensity_changed)
        self.fill_intensity.sliderReleased().connect(self._on_fill_intensity_released)
        layout.addRow("Fill intensity:", self.fill_intensity)

        # Fill light color
        self.fill_color = ColorButton()
        self.fill_color.color_changed.connect(self._on_fill_color_changed)
        layout.addRow("Fill color:", self.fill_color)

        # Ambient light intensity
        self.ambient_intensity = LabeledSlider(0, 2.0, 0.4, decimals=2)
        self.ambient_intensity.valueChanged().connect(
            self._on_ambient_intensity_changed
        )
        self.ambient_intensity.sliderReleased().connect(
            self._on_ambient_intensity_released
        )
        layout.addRow("Ambient intensity:", self.ambient_intensity)

        # Ambient light color
        self.ambient_color = ColorButton()
        self.ambient_color.color_changed.connect(self._on_ambient_color_changed)
        layout.addRow("Ambient color:", self.ambient_color)

        return tab

    def _create_shininess_tab(self):
        """Create the Shininess (material) controls tab."""
        tab = QWidget()
        tab.setAutoFillBackground(True)
        layout = QFormLayout(tab)

        # Sharpness (specular exponent)
        # Chimera range: 1-128, default 30
        self.sharpness = LabeledSlider(1, 128, 30, decimals=0)
        self.sharpness.valueChanged().connect(self._on_sharpness_changed)
        self.sharpness.sliderReleased().connect(self._on_sharpness_released)
        layout.addRow("Sharpness:", self.sharpness)

        # Reflectivity (specular reflectivity)
        # Chimera range: 0.1-10, default 1.0
        # ChimeraX default is 0.3, but we'll use similar range
        self.reflectivity = LabeledSlider(0.0, 2.0, 0.3, decimals=2)
        self.reflectivity.valueChanged().connect(self._on_reflectivity_changed)
        self.reflectivity.sliderReleased().connect(self._on_reflectivity_released)
        layout.addRow("Reflectivity:", self.reflectivity)

        return tab

    def _create_depth_cue_tab(self):
        """Create the Depth Cue controls tab."""
        tab = QWidget()
        tab.setAutoFillBackground(True)
        layout = QFormLayout(tab)

        # Enable checkbox
        self.depth_cue_enabled = QCheckBox()
        self.depth_cue_enabled.stateChanged.connect(self._on_depth_cue_enabled_changed)
        layout.addRow("Enabled:", self.depth_cue_enabled)

        # Start (0-1, default 0.5)
        self.depth_cue_start = LabeledSlider(0, 1, 0.5, decimals=2)
        self.depth_cue_start.valueChanged().connect(self._on_depth_cue_start_changed)
        self.depth_cue_start.sliderReleased().connect(self._on_depth_cue_start_released)
        layout.addRow("Start:", self.depth_cue_start)

        # End (0-1, default 1.0)
        self.depth_cue_end = LabeledSlider(0, 1, 1.0, decimals=2)
        self.depth_cue_end.valueChanged().connect(self._on_depth_cue_end_changed)
        self.depth_cue_end.sliderReleased().connect(self._on_depth_cue_end_released)
        layout.addRow("End:", self.depth_cue_end)

        # Depth cue color
        self.depth_cue_color = ColorButton()
        self.depth_cue_color.color_changed.connect(self._on_depth_cue_color_changed)
        layout.addRow("Color:", self.depth_cue_color)

        return tab

    def _sync_from_session(self):
        """Sync UI controls with current session lighting/material state."""
        v = self.session.main_view
        lp = v.lighting
        mat = v.material

        # Block signals during sync
        self._block_signals(True)

        # Lighting
        self.key_intensity.setValue(lp.key_light_intensity)
        self.fill_intensity.setValue(lp.fill_light_intensity)
        self.ambient_intensity.setValue(lp.ambient_light_intensity)

        # Colors (0-255 for ColorButton)
        self.key_color.color = tuple(int(c * 255) for c in lp.key_light_color)
        self.fill_color.color = tuple(int(c * 255) for c in lp.fill_light_color)
        self.ambient_color.color = tuple(int(c * 255) for c in lp.ambient_light_color)

        # Material
        self.sharpness.setValue(mat.specular_exponent)
        self.reflectivity.setValue(mat.specular_reflectivity)

        # Depth cue
        self.depth_cue_enabled.setChecked(lp.depth_cue)
        self.depth_cue_start.setValue(lp.depth_cue_start)
        self.depth_cue_end.setValue(lp.depth_cue_end)
        self.depth_cue_color.color = tuple(int(c * 255) for c in lp.depth_cue_color)

        self._block_signals(False)

        # Sync preview
        self._sync_preview()

    def _block_signals(self, block):
        """Block or unblock signals from all controls."""
        self.key_intensity.slider.blockSignals(block)
        self.fill_intensity.slider.blockSignals(block)
        self.ambient_intensity.slider.blockSignals(block)
        self.sharpness.slider.blockSignals(block)
        self.reflectivity.slider.blockSignals(block)
        self.depth_cue_enabled.blockSignals(block)
        self.depth_cue_start.slider.blockSignals(block)
        self.depth_cue_end.slider.blockSignals(block)

    def _sync_preview(self):
        """Sync the preview sphere with main view lighting."""
        main_lp = self.session.main_view.lighting
        main_mat = self.session.main_view.material
        preview = self.preview_widget

        # Copy lighting parameters
        lp = preview.view.lighting
        lp.key_light_intensity = main_lp.key_light_intensity
        lp.key_light_direction = main_lp.key_light_direction
        lp.key_light_color = main_lp.key_light_color
        lp.fill_light_intensity = main_lp.fill_light_intensity
        lp.fill_light_direction = main_lp.fill_light_direction
        lp.fill_light_color = main_lp.fill_light_color
        lp.ambient_light_intensity = main_lp.ambient_light_intensity
        lp.ambient_light_color = main_lp.ambient_light_color

        # Copy material parameters
        mat = preview.view.material
        mat.specular_exponent = main_mat.specular_exponent
        mat.specular_reflectivity = main_mat.specular_reflectivity

        # Update arrow directions
        preview.update_arrow_directions()

        preview.view.update_lighting = True
        preview.render()

    # === Lighting callbacks ===

    def _on_preset_changed(self, preset):
        run(self.session, f"lighting {preset}")
        self._sync_from_session()

    def _on_key_intensity_changed(self, value):
        intensity = value / self.key_intensity.scale
        self.session.main_view.lighting.key_light_intensity = intensity
        self.session.main_view.update_lighting = True
        self.session.main_view.redraw_needed = True
        self._sync_preview()

    def _on_key_intensity_released(self):
        run(self.session, f"lighting intensity {self.key_intensity.value()}")

    def _on_key_color_changed(self, color):
        run(self.session, f"lighting color {color[0]},{color[1]},{color[2]}")
        self._sync_preview()

    def _on_fill_intensity_changed(self, value):
        intensity = value / self.fill_intensity.scale
        self.session.main_view.lighting.fill_light_intensity = intensity
        self.session.main_view.update_lighting = True
        self.session.main_view.redraw_needed = True
        self._sync_preview()

    def _on_fill_intensity_released(self):
        run(self.session, f"lighting fillIntensity {self.fill_intensity.value()}")

    def _on_fill_color_changed(self, color):
        run(self.session, f"lighting fillColor {color[0]},{color[1]},{color[2]}")
        self._sync_preview()

    def _on_ambient_intensity_changed(self, value):
        intensity = value / self.ambient_intensity.scale
        self.session.main_view.lighting.ambient_light_intensity = intensity
        self.session.main_view.update_lighting = True
        self.session.main_view.redraw_needed = True
        self._sync_preview()

    def _on_ambient_intensity_released(self):
        run(self.session, f"lighting ambientIntensity {self.ambient_intensity.value()}")

    def _on_ambient_color_changed(self, color):
        run(self.session, f"lighting ambientColor {color[0]},{color[1]},{color[2]}")
        self._sync_preview()

    # === Shininess callbacks ===

    def _on_sharpness_changed(self, value):
        exponent = value / self.sharpness.scale
        self.session.main_view.material.specular_exponent = exponent
        self.session.main_view.update_lighting = True
        self.session.main_view.redraw_needed = True
        self._sync_preview()

    def _on_sharpness_released(self):
        run(self.session, f"material specularExponent {self.sharpness.value()}")

    def _on_reflectivity_changed(self, value):
        reflectivity = value / self.reflectivity.scale
        self.session.main_view.material.specular_reflectivity = reflectivity
        self.session.main_view.update_lighting = True
        self.session.main_view.redraw_needed = True
        self._sync_preview()

    def _on_reflectivity_released(self):
        run(self.session, f"material reflectivity {self.reflectivity.value()}")

    # === Depth Cue callbacks ===

    def _on_depth_cue_enabled_changed(self, state):
        enabled = state == Qt.CheckState.Checked.value
        run(self.session, f"lighting depthCue {'true' if enabled else 'false'}")

    def _on_depth_cue_start_changed(self, value):
        start = value / self.depth_cue_start.scale
        self.session.main_view.lighting.depth_cue_start = start
        self.session.main_view.redraw_needed = True

    def _on_depth_cue_start_released(self):
        run(self.session, f"lighting depthCueStart {self.depth_cue_start.value()}")

    def _on_depth_cue_end_changed(self, value):
        end = value / self.depth_cue_end.scale
        self.session.main_view.lighting.depth_cue_end = end
        self.session.main_view.redraw_needed = True

    def _on_depth_cue_end_released(self):
        run(self.session, f"lighting depthCueEnd {self.depth_cue_end.value()}")

    def _on_depth_cue_color_changed(self, color):
        run(self.session, f"lighting depthCueColor {color[0]},{color[1]},{color[2]}")

    def delete(self):
        self.preview_widget.close()
        super().delete()


class LightingPreviewWidget(QWindow):
    """OpenGL widget displaying a sphere with current lighting for preview."""

    # Colors for light arrows (RGBA 0-255)
    KEY_LIGHT_COLOR = (255, 200, 50, 255)  # Yellow/gold for key
    FILL_LIGHT_COLOR = (100, 150, 255, 255)  # Blue for fill

    def __init__(self, parent, session, lighting_gui):
        QWindow.__init__(self)
        self.session = session
        self.lighting_gui = lighting_gui
        self.widget = QWidget.createWindowContainer(self, parent)
        self.widget.setMinimumSize(360, 360)
        self.setSurfaceType(QSurface.SurfaceType.OpenGLSurface)

        # Create sphere drawing
        self.sphere = SphereDrawing()

        # Create light direction arrows
        self.key_arrow = LightArrowDrawing("key_light", self.KEY_LIGHT_COLOR)
        self.fill_arrow = LightArrowDrawing("fill_light", self.FILL_LIGHT_COLOR)

        # Add arrows as children of sphere
        self.sphere.add_drawing(self.key_arrow)
        self.sphere.add_drawing(self.fill_arrow)

        # Create view with sphere
        self.view = View(drawing=self.sphere, window_size=(360, 360))
        self.view.initialize_rendering(session.main_view.render.opengl_context)
        self.view.background_color = (40, 40, 40, 255)
        # Zoom out enough to show sphere (radius 1) plus arrows with comfortable margin
        self.view.camera.position = Place(origin=(0, 0, 6))

        # Initialize arrow directions
        self.update_arrow_directions()

        self.main_view = session.main_view

        # Mouse dragging state
        self._dragging = None  # 'key' or 'fill' or None
        self._last_pos = None

        # Enable multitouch on macOS
        if sys.platform == "darwin":
            from chimerax.core import _mac_util

            nsview_pointer = int(self.winId())
            _mac_util.enable_multitouch(nsview_pointer)

        # Show the window
        self.show()

        # Track whether we need to render
        self._needs_render = True

    def update_arrow_directions(self):
        """Update arrow positions from current lighting."""
        lp = self.session.main_view.lighting
        self.key_arrow.direction = lp.key_light_direction
        self.fill_arrow.direction = lp.fill_light_direction

    def close(self):
        self.sphere.delete()
        self.view.delete()
        QWindow.destroy(self)

    def exposeEvent(self, event):
        """Render when window is exposed."""
        if self.isExposed() and not self.session.update_loop.blocked():
            self._needs_render = False
            self._do_render()

    def resizeEvent(self, event):
        size = event.size()
        self.view.resize(size.width(), size.height())
        self._needs_render = True
        self.requestUpdate()

    def mousePressEvent(self, event):
        """Handle mouse press - check if clicking on an arrow."""
        if event.button() != Qt.MouseButton.LeftButton:
            return

        x, y = event.position().x(), event.position().y()
        pick = self._pick_arrow(x, y)
        if pick:
            self._dragging = pick
            self._last_pos = (x, y)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        """Handle mouse drag to rotate light direction."""
        if self._dragging is None:
            # Check for hover
            x, y = event.position().x(), event.position().y()
            pick = self._pick_arrow(x, y)
            if pick:
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)
            return

        x, y = event.position().x(), event.position().y()
        if self._last_pos is None:
            self._last_pos = (x, y)
            return

        # Calculate rotation from mouse movement
        dx = x - self._last_pos[0]
        dy = y - self._last_pos[1]
        self._last_pos = (x, y)

        if abs(dx) < 0.5 and abs(dy) < 0.5:
            return

        # Get current direction
        lp = self.session.main_view.lighting
        if self._dragging == "key":
            direction = np.array(lp.key_light_direction, dtype=np.float32)
        else:
            direction = np.array(lp.fill_light_direction, dtype=np.float32)

        # Rotate direction based on mouse movement
        # Use camera orientation to determine rotation axes
        cam_pos = self.view.camera.position
        right = cam_pos.axes()[0]  # Camera right vector
        up = cam_pos.axes()[1]  # Camera up vector

        # Sensitivity: degrees of rotation per pixel of mouse movement
        sensitivity = 0.5

        # Rotate around up axis for horizontal mouse movement
        # Rotate around right axis for vertical mouse movement
        from chimerax.geometry import rotation as make_rotation

        rot_h = make_rotation(up, dx * sensitivity)
        rot_v = make_rotation(right, dy * sensitivity)

        # Apply rotations
        direction = rot_h.transform_vector(direction)
        direction = rot_v.transform_vector(direction)
        direction = direction / np.linalg.norm(direction)

        # Update lighting
        if self._dragging == "key":
            lp.key_light_direction = direction
        else:
            lp.fill_light_direction = direction

        self.session.main_view.update_lighting = True
        self.session.main_view.redraw_needed = True

        # Update arrows and sync GUI
        self.update_arrow_directions()
        self.view.update_lighting = True
        self.render()
        self.lighting_gui._sync_from_session()

    def mouseReleaseEvent(self, event):
        """Handle mouse release."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = None
            self._last_pos = None
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def _pick_arrow(self, x, y):
        """Check if position hits an arrow. Returns 'key', 'fill', or None."""
        # Convert to view coordinates
        width, height = self.view.window_size
        if width <= 0 or height <= 0:
            return None

        # Get ray from camera through pixel
        camera = self.view.camera
        # Normalized device coordinates
        nx = (2 * x / width) - 1
        ny = 1 - (2 * y / height)

        # Ray origin and direction in world space
        cam_pos = camera.position
        origin = cam_pos.origin()

        # For perspective camera, ray goes through pixel
        # Field of view calculation
        fov = camera.field_of_view
        aspect = width / height
        tan_fov = np.tan(np.radians(fov / 2))

        # Direction in camera space
        dir_cam = np.array([nx * tan_fov * aspect, ny * tan_fov, -1], dtype=np.float32)
        dir_cam = dir_cam / np.linalg.norm(dir_cam)

        # Transform to world space
        direction = cam_pos.transform_vector(dir_cam)

        # Check intersection with arrows (simplified - check distance to arrow axis)
        for arrow, name in [(self.key_arrow, "key"), (self.fill_arrow, "fill")]:
            # Arrow center position (arrow is positioned outside sphere pointing in)
            arrow_center = -arrow.direction * (
                arrow.sphere_radius + arrow.arrow_length / 2
            )
            # Simple distance check (approximate)
            to_arrow = arrow_center - origin
            t = np.dot(to_arrow, direction)
            if t > 0:
                closest = origin + t * direction
                dist = np.linalg.norm(closest - arrow_center)
                if dist < 0.25:  # Hit threshold
                    return name

        return None

    def render(self):
        """Request a render of the preview."""
        self._needs_render = True
        self.requestUpdate()

    def event(self, event):
        """Handle update request events."""
        from Qt.QtCore import QEvent

        if event.type() == QEvent.Type.UpdateRequest and self._needs_render:
            self._needs_render = False
            if self.isExposed():
                self._do_render()
            return True
        return super().event(event)

    def _do_render(self):
        """Actually render the preview."""
        ww, wh = self.main_view.window_size
        width, height = self.view.window_size
        if ww <= 0 or wh <= 0 or width <= 0 or height <= 0:
            return
        if self.view is None or self.view.render is None:
            return

        mvwin = self.view.render.use_shared_context(self)
        try:
            self.view.draw()
        except Exception:
            pass
        finally:
            self.main_view.render.use_shared_context(mvwin)
        self.view.render.done_current()
