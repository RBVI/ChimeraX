import sys

from enum import StrEnum
from Qt.QtGui import QMouseEvent

from Qt.QtCore import Qt
from Qt.QtGui import QWindow, QSurface
from Qt.QtWidgets import (
    QHBoxLayout, QVBoxLayout, QWidget, QLabel
    , QComboBox, QSlider, QSizePolicy
)

from chimerax.core.colors import Color
from chimerax.core.tools import ToolInstance
from chimerax.core.commands import log_equivalent_command
from chimerax.std_commands.lighting import lighting, lighting_model, lighting_settings
from chimerax.ui import MainToolWindow
from chimerax.ui.widgets import ColorButton
from chimerax.geometry import Place
from chimerax.graphics import View, MonoCamera, Drawing

class LightingMode(StrEnum):
    Simple = "simple"
    Soft = "soft"
    Full = "full"
    Flat = "flat"
    Gentle = "gentle"

class SphereDrawing(Drawing):
    def __init__(self, facets: int = 8192):
        super().__init__("lightingsphere")
        from chimerax.surface import sphere_geometry2
        self.set_geometry(*sphere_geometry2(facets))

class LightingGUI(ToolInstance):

    help = "help:user/tools/lightinggui.html"

    def __init__(self, session = None, tool_name = "Lighting GUI"):
        super().__init__(session, tool_name)
        self.tool_window = MainToolWindow(self)
        parent = self.tool_window.ui_area
        parent.setLayout(QHBoxLayout())
        self.menu_container = QWidget()
        self.menu_container.setLayout(QVBoxLayout())
        self.preset_spinbox = QComboBox()
        self.preset_spinbox.addItems([mode for mode in LightingMode])
        self.ambient_light_intensity_slider = QSlider()
        self.fill_light_intensity_slider = QSlider()
        self.key_light_intensity_slider = QSlider()
        self.ambient_light_intensity_slider.setOrientation(Qt.Orientation.Horizontal)
        self.fill_light_intensity_slider.setOrientation(Qt.Orientation.Horizontal)
        self.key_light_intensity_slider.setOrientation(Qt.Orientation.Horizontal)
        self.ambient_light_intensity_slider.setRange(0, 100)
        self.fill_light_intensity_slider.setRange(0, 100)
        self.key_light_intensity_slider.setRange(0, 100)
        self.ambient_light_intensity_slider.valueChanged.connect(self._on_ambient_light_intensity_slider_changed)
        self.fill_light_intensity_slider.valueChanged.connect(self._on_fill_light_intensity_slider_changed)
        self.key_light_intensity_slider.valueChanged.connect(self._on_key_light_intensity_slider_changed)
        self.key_light_color_button = ColorButton()
        self.fill_light_color_button = ColorButton()
        self.ambient_light_color_button = ColorButton()
        self.key_light_color_button.color_changed.connect(self._on_key_light_color_changed)
        self.fill_light_color_button.color_changed.connect(self._on_fill_light_color_changed)
        self.ambient_light_color_button.color_changed.connect(self._on_ambient_light_color_changed)
        self.lighting_widget = LightingDirectionWidget(parent, session)
        parent.layout().addWidget(self.menu_container, 1)
        parent.layout().addWidget(self.lighting_widget.widget, 1)
        self.menu_container.layout().addWidget(self.preset_spinbox)
        self.menu_container.layout().addWidget(QLabel("Ambient Light Intensity"))
        self.menu_container.layout().addWidget(self.ambient_light_intensity_slider)
        self.menu_container.layout().addWidget(QLabel("Fill Light Intensity"))
        self.menu_container.layout().addWidget(self.fill_light_intensity_slider)
        self.menu_container.layout().addWidget(QLabel("Key Light Intensity"))
        self.menu_container.layout().addWidget(self.key_light_intensity_slider)
        self.menu_container.layout().addWidget(QLabel("Key Light Color"))
        self.menu_container.layout().addWidget(self.key_light_color_button)
        self.menu_container.layout().addWidget(QLabel("Fill Light Color"))
        self.menu_container.layout().addWidget(self.fill_light_color_button)
        self.menu_container.layout().addWidget(QLabel("Ambient Light Color"))
        self.menu_container.layout().addWidget(self.ambient_light_color_button)
        self.menu_container.layout().addStretch()
        self.preset_spinbox.currentTextChanged.connect(self._set_lighting_preset)
        self.tool_window.manage()

    def _set_lighting_preset(self, preset):
        log_equivalent_command(self.session, f'lighting %s' % preset)
        lighting(self.session, preset=preset)
        self.lighting_widget.apply_lighting_preset(LightingMode(preset))

    def _on_ambient_light_intensity_slider_changed(self, intensity):
        log_equivalent_command(self.session, f'lighting ambient_intensity %f' % (intensity / 100))
        lighting(self.session, ambient_intensity=intensity / 100)
        self.lighting_widget.set_ambient_intensity(intensity)

    def _on_fill_light_intensity_slider_changed(self, intensity):
        log_equivalent_command(self.session, f'lighting fill_intensity %f' % (intensity / 100))
        lighting(self.session, fill_intensity=intensity / 100)
        self.lighting_widget.set_fill_intensity(intensity)

    def _on_key_light_intensity_slider_changed(self, intensity):
        # The lighting command does not take an intensity parameter for the key light
        self.session.main_view.lighting.key_light_intensity = intensity / 100
        self.session.main_view.update_lighting = True
        self.session.main_view.redraw_needed = True
        self.lighting_widget.set_key_intensity(intensity)

    def _on_key_light_color_changed(self, color):
        _color = Color((color[0], color[1], color[2]), limit = False)
        from chimerax.core.commands import run
        run(self.session, f'lighting color %i,%i,%i' % (color[0], color[1], color[2]))
        #lighting(self.session, color = _color)
        self.lighting_widget.set_key_color(_color)

    def _on_fill_light_color_changed(self, color):
        _color = Color((color[0], color[1], color[2]), limit = False)
        log_equivalent_command(self.session, f'lighting fill_color %i,%i,%i' % (color[0], color[1], color[2]))
        lighting(self.session, fill_color = _color)
        self.lighting_widget.set_fill_color(_color)

    def _on_ambient_light_color_changed(self, color):
        _color = Color((color[0], color[1], color[2]), limit = False)
        log_equivalent_command(self.session, f'lighting ambient_color %i,%i,%i' % (color[0], color[1], color[2]))
        lighting(self.session, ambient_color = _color)
        self.lighting_widget.set_ambient_color(_color)


    def delete(self):
        self.lighting_widget.close()

class LightingDirectionWidget(QWindow):
    """Display a sphere with two arrows that shows the direction of lights"""

    def __init__(self, parent, session):
        #lighting = session.ui.main_window.main_view.lighting
        QWindow.__init__(self)
        self.session = session
        self.widget = QWidget.createWindowContainer(self, parent)
        self.setSurfaceType(QSurface.SurfaceType.OpenGLSurface)
        self.view = View(drawing = SphereDrawing(), window_size=(0, 0))
        self.view.initialize_rendering(session.main_view.render.opengl_context)
        self.main_view = self.session.main_view
        self.view.background_color = (0, 0, 0, 255)
        self.view.camera.position = Place(origin=(0,0,5))
        if sys.platform == "darwin":
            from chimerax.core import _mac_util
            nsview_pointer = int(self.winId())
            _mac_util.enable_multitouch(nsview_pointer)
            self.widget.touchEvent = self.touchEvent
        self.handler = session.triggers.add_handler('frame drawn', self.render)

    def close(self):
        self.view.drawing.delete()
        self.view.delete()
        self.session.triggers.remove_handler(self.handler)
        QWindow.destroy(self)

    def resizeEvent(self, event):  # noqa
        size = event.size()
        width = size.width()
        height = size.height()
        self.set_viewport(width, height)

    def set_viewport(self, width, height):
        # Don't need make_current, since OpenGL isn't used
        # until rendering
        self.view.resize(width, height)

    def render(self, *args, **kwargs):
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
            self.view.draw()
        except Exception as e: # noqa
            # This line is here so you can set a breakpoint on it and figure out what's going wrong
            # because ChimeraX's interface will not tell you.
            pass
        finally:
            # Target opengl context back to main graphics window.
            self.main_view.render.use_shared_context(mvwin)
        self.view.render.done_current()

    def mousePressEvent(self, a0: QMouseEvent) -> None:
        # Determine whether an arrow is under the mouse
        # Set the tracked arrow to that arrow, or None
        pass

    def mouseMoveEvent(self, event):
        # Otherwise do nothing
        if self._tracked_arrow:
            # Move the arrow and adjust the lighting accordingly
            pass
        else:
            pass

    def mouseReleaseEvent(self):
        self._tracked_arrow = None

    def _add_arrow(self):
        pass

    def _remove_arrow(self):
        pass

    #region Responding to UI Switches, Drags, ComboBoxes...
    def apply_lighting_preset(self, preset: LightingMode):
        lighting = self.view.lighting
        silhouette = self.view.silhouette
        ms_directions = lighting_settings(self.session).lighting_multishadow_directions
        if preset == LightingMode.Flat:
            lighting.shadows = False
            lighting.multishadow = 0
            lighting.key_light_intensity = 0
            lighting.fill_light_intensity = 0
            lighting.ambient_light_intensity = 1.45
            silhouette.enabled = True
            silhouette.depth_jump = 0.01
        elif preset == LightingMode.Simple:
            lighting.shadows = False
            lighting.multishadow = 0
            silhouette.depth_jump = 0.03
            lighting.set_default_parameters(self.view.background_color)
        elif preset == LightingMode.Soft:
            lighting.shadows = False
            lighting.multishadow = ms_directions
            lighting.key_light_intensity = 0
            lighting.fill_light_intensity = 0
            lighting.ambient_light_intensity = 1.5
            lighting.multishadow_depth_bias = 0.01
            lighting.multishadow_map_size = 1024
            silhouette.depth_jump = 0.03
        elif preset == LightingMode.Full:
            lighting.shadows = True
            lighting.multishadow = ms_directions
            lighting.key_light_intensity = 0.7
            lighting.fill_light_intensity = 0.3
            lighting.ambient_light_intensity = 0.8
            lighting.multishadow_depth_bias = 0.01
            lighting.multishadow_map_size = 1024
            silhouette.depth_jump = 0.03
        elif preset == LightingMode.Gentle:
            lighting.shadows = False
            lighting.multishadow = ms_directions
            lighting.key_light_intensity = 0
            lighting.fill_light_intensity = 0
            lighting.ambient_light_intensity = 1.5
            lighting.multishadow_depth_bias = 0.05
            lighting.multishadow_map_size = 128
            silhouette.depth_jump = 0.03
        self.view.update_lighting = True

    def set_ambient_intensity(self, intensity):
        self.view.lighting.ambient_light_intensity = intensity / 100
        self.view.update_lighting = True

    def set_fill_intensity(self, intensity):
        self.view.lighting.fill_light_intensity = intensity / 100
        self.view.update_lighting = True

    def set_key_intensity(self, intensity):
        self.view.lighting.key_light_intensity = intensity / 100
        self.view.update_lighting = True

    def set_key_color(self, color: Color):
        self.view.lighting.key_light_color = color.rgba[:3]
        self.view.update_lighting = True

    def set_fill_color(self, color: Color):
        self.view.lighting.fill_light_color = color.rgba[:3]
        self.view.update_lighting = True

    def set_ambient_color(self, color: Color):
        self.view.lighting.ambient_light_color = color.rgba[:3]
        self.view.update_lighting = True

    #endregion
