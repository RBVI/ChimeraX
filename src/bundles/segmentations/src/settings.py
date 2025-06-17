from chimerax.core.settings import Settings

from chimerax.segmentations.actions import (
    MouseAction,
    HandAction,
    Handedness,
)

_seg_bundle_settings = None


class _SegmentationSettings(Settings):
    EXPLICIT_SAVE = {
        "start_vr_automatically": False,
        "set_mouse_modes_automatically": False,
        "set_hand_modes_automatically": False,
        "display_guidelines": False,
        "display_color_keys": False,
        "automatically_switch_models_on_menu_changes": False,
        "default_view": 0,  # 4 x 4
        "default_segmentation_opacity": 80,  # %
        "mouse_3d_right_click": MouseAction.ADD_TO_SEGMENTATION,
        "mouse_3d_middle_click": MouseAction.MOVE_SPHERE,
        "mouse_3d_scroll": MouseAction.RESIZE_SPHERE,
        "mouse_3d_left_click": MouseAction.NONE,
        "vr_thumbstick": HandAction.RESIZE_CURSOR,
        "vr_trigger": HandAction.ADD_TO_SEGMENTATION,
        "vr_grip": HandAction.MOVE_CURSOR,
        "vr_a_button": HandAction.ERASE_FROM_SEGMENTATION,
        "vr_b_button": HandAction.NONE,
        "vr_handedness": Handedness.RIGHT,
    }


def get_settings(session):
    global _seg_bundle_settings
    if _seg_bundle_settings is None:
        _seg_bundle_settings = _SegmentationSettings(session, "Segmentation Tool")
    return _seg_bundle_settings
