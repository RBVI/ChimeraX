import string

from enum import StrEnum, Enum

from chimerax.core.settings import Settings

_graphics_bundle_settings = None


class GraphicsSetting(StrEnum):
    CHECK_SHADER_UNIFORMS = "check_shader_uniforms"
    DEPTH_PEELING = "depth_peeling"
    DEPTH_PEELING_LAYERS = "depth_peeling_layers"
    BACKGROUND_COLOR = "background_rbga"
    HIGHLIGHT_COLOR = "highlight_color"
    HIGHLIGHT_WIDTH = "highlight_width"
    SHOW_DEBUG_MENU = "show_debug_menu"
    VSYNC = "vertical_sync"
    MAX_FRAMERATE = "max_framerate"
    SHOW_FRAMERATE = "show_framerate"
    KEY_LIGHT_DIRECTION = "key_light_direction"
    KEY_LIGHT_INTENSITY = "key_light_intensity"
    KEY_LIGHT_COLOR = "key_light_color"
    FILL_LIGHT_DIRECTION = "fill_light_direction"
    FILL_LIGHT_INTENSITY = "fill_light_intensity"
    FILL_LIGHT_COLOR = "fill_light_color"
    AMBIENT_LIGHT_DIRECTION = "ambient_light_direction"
    AMBIENT_LIGHT_INTENSITY = "ambient_light_intensity"
    AMBIENT_LIGHT_COLOR = "ambient_light_color"
    SILHOUETTES = "silhouettes"


def setting_display_name(setting: GraphicsSetting) -> str:
    if setting == GraphicsSetting.VSYNC:
        return "V-Sync"
    return string.capwords(" ".join(setting.value.split("_")))


class RotationMethod(Enum):
    FIXED = 0
    FRONT_CENTER = 1
    CENTER_OF_VIEW = 2


class GraphicsSettingCategory(StrEnum):
    DEBUGGING = "Debugging"
    GENERAL = "General"
    TRANSPARENCY = "Transparency"
    DEPICTION = "Depiction"
    LIGHTING = "Lighting"


class _GraphicsSettings(Settings):
    EXPLICIT_SAVE = {
        GraphicsSetting.CHECK_SHADER_UNIFORMS: True,
        GraphicsSetting.DEPTH_PEELING: False,
        GraphicsSetting.DEPTH_PEELING_LAYERS: 8,
        GraphicsSetting.SHOW_DEBUG_MENU: False,
        GraphicsSetting.SHOW_FRAMERATE: False,
        GraphicsSetting.VSYNC: False,
        GraphicsSetting.MAX_FRAMERATE: 60,
        GraphicsSetting.SILHOUETTES: False,
    }


def get_graphics_settings(session):
    global _graphics_bundle_settings
    if _graphics_bundle_settings is None:
        _graphics_bundle_settings = _GraphicsSettings(session, "Graphics Bundle")
    return _graphics_bundle_settings
