# vim: set expandtab ts=4 sw=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===
import os

from enum import Enum

from chimerax.core.commands import run
from chimerax.core.settings import Settings
from chimerax.mouse_modes import MouseMode
from chimerax.geometry.place import Place

from chimerax.segmentations.ui import find_segmentation_tool
from chimerax.segmentations.triggers import activate_trigger, Trigger

_saved_mouse_bindings = {
    "left": {
        "none": None,
        "shift": None,
        "ctrl": None,
        "command": None,
        "alt": None,
    },
    "right": {
        "none": None,
        "shift": None,
        "ctrl": None,
        "command": None,
        "alt": None,
    },
    "middle": {
        "none": None,
        "shift": None,
        "ctrl": None,
        "command": None,
        "alt": None,
    },
    "wheel": {
        "none": None,
        "shift": None,
        "ctrl": None,
        "command": None,
        "alt": None,
    },
    "pause": {
        "none": None,
        "shift": None,
        "ctrl": None,
        "command": None,
        "alt": None,
    },
}

_saved_hand_bindings = {
    "trigger": None,
    "grip": None,
    "touchpad": None,
    "thumbstick": None,
    "menu": None,
    "a": None,
    "b": None,
    "x": None,
    "y": None,
}

_have_saved_mouse_bindings = False
_have_saved_hand_bindings = False


def mouse_bindings_saved():
    global _have_saved_mouse_bindings
    return _have_saved_mouse_bindings


def hand_bindings_saved():
    global _have_saved_mouse_bindings
    return _have_saved_mouse_bindings


def save_mouse_bindings(session):
    global _saved_mouse_bindings
    global _have_saved_mouse_bindings
    for binding in session.ui.mouse_modes.bindings:
        if not binding.modifiers:
            _saved_mouse_bindings[binding.button]["none"] = binding.mode.name
        else:
            for modifier in binding.modifiers:
                _saved_mouse_bindings[binding.button][modifier] = binding.mode.name
    _have_saved_mouse_bindings = True
    activate_trigger(Trigger.MouseModesChanged, _have_saved_mouse_bindings)


def restore_mouse_bindings(session):
    global _saved_mouse_bindings
    global _have_saved_mouse_bindings
    if _have_saved_mouse_bindings:
        run(
            session,
            (
                "ui mousemode shift wheel '"
                + _saved_mouse_bindings["wheel"]["shift"]
                + "'"
                if _saved_mouse_bindings["wheel"]["shift"]
                else "ui mousemode shift wheel 'none'"
            ),
        )
        run(
            session,
            (
                "ui mousemode right '" + _saved_mouse_bindings["right"]["none"] + "'"
                if _saved_mouse_bindings["right"]["none"]
                else "ui mousemode right 'none'"
            ),
        )
        run(
            session,
            (
                "ui mousemode shift right '"
                + _saved_mouse_bindings["right"]["shift"]
                + "'"
                if _saved_mouse_bindings["right"]["shift"]
                else "ui mousemode shift right 'none'"
            ),
        )
        run(
            session,
            (
                "ui mousemode shift middle '"
                + _saved_mouse_bindings["middle"]["shift"]
                + "'"
                if _saved_mouse_bindings["middle"]["shift"]
                else "ui mousemode shift middle 'none'"
            ),
        )
        _have_saved_mouse_bindings = False
        activate_trigger(Trigger.MouseModesChanged, _have_saved_mouse_bindings)
    else:
        session.logger.warning("No mouse bindings saved")


def save_hand_bindings(session, handedness):
    global _saved_hand_bindings
    global _have_saved_hand_bindings
    from chimerax.vive.vr import SteamVRCamera
    from chimerax.vive.xr import OpenXRCamera
    from chimerax.vive.vr import vr_camera as steamvr_camera
    from chimerax.vive.vr import vr_button as steamvr_button
    from chimerax.vive.xr import vr_camera as openxr_camera
    from chimerax.vive.xr import vr_button as openxr_button
    if type(session.main_view.camera) is SteamVRCamera:
        vr_camera = steamvr_camera
        vr_button = steamvr_button
        from openvr import (
            k_EButton_Grip as grip,
            k_EButton_ApplicationMenu as menu,
            k_EButton_SteamVR_Trigger as trigger,
            k_EButton_SteamVR_Touchpad as touchpad,
            k_EButton_A as a,
        )

        button_names = {
            grip: "grip",
            menu: "menu",
            trigger: "trigger",
            touchpad: "thumbstick",
            a: "a",
        }
        c = vr_camera(session)
        hclist = [hc for hc in c.hand_controllers() if hc._side == str(handedness)]
        if not hclist:
            ...  # error
        hc = hclist[0]
        for button, binding in hc._modes.items():
            _saved_hand_bindings[button_names[button]] = binding.name
        _have_saved_hand_bindings = True
        return True
    elif type(session.main_view.camera) is OpenXRCamera:
        # TODO
        vr_camera = openxr_camera
        vr_button = openxr_button
        _have_saved_hand_bindings = True
        return True
    return False


def restore_hand_bindings(session):
    global _saved_hand_bindings
    global _have_saved_hand_bindings
    from chimerax.vive.vr import SteamVRCamera
    from chimerax.vive.xr import OpenXRCamera
    camera = session.main_view.camera
    if isinstance(camera, SteamVRCamera) or isinstance(camera, OpenXRCamera):
        if _have_saved_hand_bindings:
            run(session, f'vr button trigger {_saved_hand_bindings["trigger"]}')
            run(
                session,
                f'vr button thumbstick {_saved_hand_bindings["thumbstick"]}',
            )
            run(session, f'vr button grip {_saved_hand_bindings["grip"]}')
            run(session, f'vr button a {_saved_hand_bindings["a"]}')
            run(session, f'vr button b {_saved_hand_bindings["b"]}')
            run(session, f'vr button x {_saved_hand_bindings["x"]}')
            _have_saved_mouse_bindings = False
            return True
        else:
            session.logger.warning("No hand bindings saved")
            return True
    return False


class CreateSegmentation3DMouseMode(MouseMode):
    """Use the segmentation sphere to mark off regions of data in 3D."""

    name = "create segmentations"
    icon_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "icons",
        "create_segmentation.png",
    )

    def mouse_down(self, event):
        MouseMode.mouse_down(self, event)
        activate_trigger(Trigger.SegmentationStarted, 1)
        activate_trigger(Trigger.SegmentationMouseModeMoveEvent, (*self.mouse_motion(event), event.shift_down(), 1))

    def mouse_drag(self, event):
        activate_trigger(Trigger.SegmentationMouseModeMoveEvent, (*self.mouse_motion(event), event.shift_down(), 1))

    def mouse_up(self, event):
        MouseMode.mouse_up(self, event)
        activate_trigger(Trigger.SegmentationMouseModeMoveEvent, (*self.mouse_motion(event), event.shift_down(), 1))
        activate_trigger(Trigger.SegmentationEnded, 1)

    def wheel(self, event):
        activate_trigger(Trigger.SegmentationMouseModeWheelEvent, event.wheel_value())

    def vr_press(self, event):
        activate_trigger(Trigger.SegmentationStarted, 1)
        activate_trigger(Trigger.SegmentationMouseModeVRMoveEvent, (Place(), event.shift_down(), 1))

    def vr_release(self, event):
        MouseMode.mouse_up(self, event)
        activate_trigger(Trigger.SegmentationMouseModeVRMoveEvent, (Place(), event.shift_down(), 1))
        activate_trigger(Trigger.SegmentationEnded, 1)

    def vr_motion(self, event):
        activate_trigger(Trigger.SegmentationMouseModeVRMoveEvent, (event.motion, 1))

class EraseSegmentation3DMouseMode(MouseMode):
    """Use the segmentation sphere to erase regions of data in 3D."""

    name = "erase segmentations"
    icon_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "icons",
        "create_segmentation.png",
    )

    def mouse_down(self, event):
        MouseMode.mouse_down(self, event)
        activate_trigger(Trigger.SegmentationStarted)
        activate_trigger(Trigger.SegmentationMouseModeMoveEvent, (*self.mouse_motion(event), event.shift_down(), 0))

    def wheel(self, event):
        activate_trigger(Trigger.SegmentationMouseModeWheelEvent, event.wheel_value())

    def mouse_drag(self, event):
        activate_trigger(Trigger.SegmentationMouseModeMoveEvent, (*self.mouse_motion(event), event.shift_down(), 0))

    def mouse_up(self, event):
        MouseMode.mouse_up(self, event)
        activate_trigger(Trigger.SegmentationMouseModeMoveEvent, (*self.mouse_motion(event), event.shift_down(), 0))
        activate_trigger(Trigger.SegmentationEnded, 0)

    def vr_press(self, event):
        activate_trigger(Trigger.SegmentationStarted, 0)
        activate_trigger(Trigger.SegmentationMouseModeMoveEvent, (Place(), mouse_motion(event), event.shift_down(), 0))

    def vr_release(self, event):
        MouseMode.mouse_up(self, event)
        activate_trigger(Trigger.SegmentationMouseModeMoveEvent, (Place(), event.shift_down(), 0))
        activate_trigger(Trigger.SegmentationEnded, 0)

    def vr_motion(self, event):
        activate_trigger(Trigger.SegmentationMouseModeVRMoveEvent, (event.motion, 0))

class Move3DSegmentationSphereMouseMode(MouseMode):
    name = "move segmentation cursor"
    icon_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "icons",
        "move_cursor.png",
    )

    def mouse_drag(self, event):
        activate_trigger(Trigger.SegmentationMouseModeMoveEvent, (*self.mouse_motion(event), event.shift_down(), None))

    def vr_motion(self, event):
        activate_trigger(Trigger.SegmentationMouseModeVRMoveEvent, (event.motion, 0))

class Toggle3DSegmentationVisibilityMouseMode(MouseMode):
    name = "toggle segmentation visibility"

    def vr_press(self, event):
        activate_trigger(Trigger.SegmentationVisibilityChanged, False)

    def vr_release(self, event):
        activate_trigger(Trigger.SegmentationVisibilityChanged, True)


class Resize3DSegmentationSphereMouseMode(MouseMode):
    name = "resize segmentation cursor"
    icon_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "icons",
        "resize_cursor.png",
    )

    def wheel(self, event):
        activate_trigger(Trigger.SegmentationMouseModeWheelEvent, event.wheel_value())

    def vr_motion(self, event):
        activate_trigger(Trigger.SegmentationMouseModeVRMoveEvent, (event.motion, 0))

    def vr_thumbstick(self, event):
        activate_trigger(Trigger.SegmentationMouseModeWheelEvent, event.y)
