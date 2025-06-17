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

"""\
Segmentations triggers ======================

This module wraps chimerax.core.triggerset, and provides public signals all
other code may subscribe to. Module level functions are the same as TriggerSet
functions.

SEGMENTATION_ADDED: Activated when this bundle's segmentation tracker recognizes
                    a new segmentation has been opened. May be emitted at the
                    end of this bundle's ADD_MODEL trigger handler.

SEGMENTATION_REMOVED: Activated when this bundle's segmentation tracker
                      recognizes an open segmentation has been closed. May be
                      emitted at the end of this bundle's REMOVE_MODEL trigger
                      handler.

SEGMENTATION_STARTED: Activated by a hand mode when either the create or erase
                      segmentation button is pressed

SEGMENTATION_ENDED: Activated by a hand mode when either the create or erase segmentation
                    button is release

SEGMENTATION_MOUSE_MODE_MOVE_EVENT: Activated when the spherical (3D) segmentation cursor
                                   is moved on desktop

SEGMENTATION_MODIFIED: Should be activated any time a segmentation is modified,
                       i.e. regions are added to or subtracted from the
                       segmentation.

AXIAL_CURSOR_MOVED: Activated whenever the user moves their mouse over a
                    PlaneViewer displaying an axial slice.

CORONAL_CURSOR_MOVED: Activated whenever the user moves their mouse over a
                      PlaneViewer displaying a coronal slice.

SAGITTAL_CURSOR_MOVED: Activated whenever the user moves their mouse over a
                       PlaneViewer displaying a sagittal slice.

SPHERE_CURSOR_MOVED: Activated whenever the segmentation sphere is moved by the
                     corresponding mouse or hand mode.

AXIAL_CURSOR_RESIZED: Activated whenever the user resizes the segmentation
                      cursor in a PlaneViewer displaying an axial slice.

CORONAL_CURSOR_RESIZED: Activated whenever the user resizes the segmentation
                        cursor in a PlaneViewer displaying a coronal slice.

SAGITTAL_CURSOR_RESIZED: Activated whenever the user resizes the segmentation
                         cursor in a PlaneViewer displaying a sagittal slice.

SPHERE_CURSOR_RESIZED: Activated whenever the user resizes the spherical
                       segmentation cursor using the corresponding mouse or hand
                       mode.

VIEW_LAYOUT_CHANGED: (EXPERIMENTAL) Activated when the view layout changes. May
                     be moved to chimerax.ui

GUIDELINES_VISIBILITY_CHANGED: Activated when a UI or command changes the
                               visibility of the guidelines over PlaneViewers.
                               Callers will need to set the value of
                               the display_guidelines attribute of a Segmentations
                               bundle settings object before activating.
                               Listeners should re-read this bundle's settings
                               and react accordingly to the new value.
                               If you are both a caller and a listener, take care
                               to block your own handler when activating.

HAND_MODES_CHANGED: Activated whenever the hand modes preset is toggled on or off.

MOUSE_MODES_CHANGED: Activated whenever the mouse modes preset is toggled on or off.
"""

# from collections import defaultdict
from typing import Any, Optional
from enum import StrEnum

from chimerax.core.triggerset import TriggerSet

from chimerax.segmentations.types import Axis

class Trigger(StrEnum):
    SegmentationAdded = "segmentation added"
    SegmentationRemoved = "segmentation removed"
    SegmentationModified = "segmentation modified"

    SegmentationStarted = "segmentation add started"
    SegmentationEnded = "segmentation add ended"

    SegmentationMouseModeMoveEvent = "segmentation mouse mode move event"
    SegmentationMouseModeVRMoveEvent = "segmentation mouse mode vr move event"
    SegmentationMouseModeWheelEvent = "segmentation mouse mode wheel event"
    SegmentationMouseModeVRWheelEvent = "segmentation mouse mode vr wheel event"

    SegmentationVisibilityChanged = "segmentation visibility changed"

    ReferenceModelChanged = "reference model changed"
    ActiveSegmentationChanged = "active segmentation changed"

    PlaneViewerEnter = "plane viewer enter"
    PlaneViewerLeave = "plane viewer leave"

    AxialCursorMoved = "axial cursor moved"
    CoronalCursorMoved = "coronal cursor moved"
    SagittalCursorMoved = "sagittal cursor moved"
    SphereCursorMoved = "sphere cursor moved"

    AxialCursorResized = "axial cursor resized"
    CoronalCursorResized = "coronal cursor resized"
    SagittalCursorResized = "sagittal cursor resized"
    SphereCursorResized = "sphere cursor resized"


    ViewLayoutChanged = "view layout changed"
    GuidelinesVisibilityChanged = "guidelines visibility changed"
    ColorKeysVisibilityChanged = "color keys visibility changed"

    HandModesChanged = "hand modes changed"
    MouseModesChanged = "mouse modes changed"

SEGMENTATION_ADDED = Trigger.SegmentationAdded
SEGMENTATION_REMOVED = Trigger.SegmentationRemoved
SEGMENTATION_MODIFIED = Trigger.SegmentationModified

SEGMENTATION_STARTED = Trigger.SegmentationStarted
SEGMENTATION_ENDED = Trigger.SegmentationEnded

REFERENCE_MODEL_CHANGED = Trigger.ReferenceModelChanged
ACTIVE_SEGMENTATION_CHANGED = Trigger.ActiveSegmentationChanged

AXIAL_CURSOR_MOVED = Trigger.AxialCursorMoved
CORONAL_CURSOR_MOVED = Trigger.CoronalCursorMoved
SAGITTAL_CURSOR_MOVED = Trigger.SagittalCursorMoved
SPHERE_CURSOR_MOVED = Trigger.SphereCursorMoved

AXIAL_CURSOR_RESIZED = Trigger.AxialCursorResized
CORONAL_CURSOR_RESIZED = Trigger.CoronalCursorResized
SAGITTAL_CURSOR_RESIZED = Trigger.SagittalCursorResized
SPHERE_CURSOR_RESIZED = Trigger.SphereCursorResized

VIEW_LAYOUT_CHANGED = Trigger.ViewLayoutChanged

HAND_MODES_CHANGED = Trigger.HandModesChanged
MOUSE_MODES_CHANGED = Trigger.MouseModesChanged

class SegmentationTriggerSet(TriggerSet):
    def __init__(self):
        super().__init__()
        self.add_trigger(Trigger.SegmentationAdded)
        self.add_trigger(Trigger.SegmentationRemoved)
        self.add_trigger(Trigger.SegmentationModified)
        self.add_trigger(Trigger.SegmentationVisibilityChanged)

        self.add_trigger(Trigger.SegmentationStarted)
        self.add_trigger(Trigger.SegmentationEnded)

        self.add_trigger(Trigger.SegmentationMouseModeMoveEvent)
        self.add_trigger(Trigger.SegmentationMouseModeVRMoveEvent)
        self.add_trigger(Trigger.SegmentationMouseModeWheelEvent)
        self.add_trigger(Trigger.SegmentationMouseModeVRWheelEvent)

        self.add_trigger(Trigger.ActiveSegmentationChanged)
        self.add_trigger(Trigger.ReferenceModelChanged)

        self.add_trigger(Trigger.PlaneViewerEnter)
        self.add_trigger(Trigger.PlaneViewerLeave)

        self.add_trigger(Trigger.AxialCursorMoved)
        self.add_trigger(Trigger.CoronalCursorMoved)
        self.add_trigger(Trigger.SagittalCursorMoved)
        self.add_trigger(Trigger.SphereCursorMoved)

        self.add_trigger(Trigger.AxialCursorResized)
        self.add_trigger(Trigger.CoronalCursorResized)
        self.add_trigger(Trigger.SagittalCursorResized)
        self.add_trigger(Trigger.SphereCursorResized)


        self.add_trigger(Trigger.ViewLayoutChanged)
        self.add_trigger(Trigger.GuidelinesVisibilityChanged)
        self.add_trigger(Trigger.ColorKeysVisibilityChanged)
        self.add_trigger(Trigger.HandModesChanged)
        self.add_trigger(Trigger.MouseModesChanged)


_triggers = SegmentationTriggerSet()

# TODO:
# _handlers_by_object = defaultdict(defaultdict(set).copy)

def activate_trigger(trigger: str, data: Optional[Any] = None) -> None:
    """
    Expected data by trigger:

    SEGMENTATION_REMOVED: the segmentation to be removed
    SEGMENTATION_ADDED: the segmentation about to be added
    SEGMENTATION_MODIFIED: the segmentation that has been modified

    ACTIVE_SEGMENTATION_CHANGED: the new active segmentation
    REFERENCE_MODEL_CHANGED: the new reference model (a volume)

    *_CURSOR_MOVED: the new coordinates of the axial cursor
    *_CURSOR_RESIZED: tuple(type of change[grow,shrink], resize step size(px))

    PLANE_VIEWER_ENTER: The axis of the entered viewer
    PLANE_VIEWER_LEAVE: The axis of the left viewer

    VIEW_LAYOUT_CHANGED: the name of the new view layout
    GUIDELINES_TOGGLED: bool representing whether guidelines are on or off
    """
    global _triggers
    _triggers.activate_trigger(trigger, data)

add_dependency = _triggers.add_dependency
add_handler = _triggers.add_handler
add_trigger = _triggers.add_trigger
block = _triggers.block
block_trigger = _triggers.block_trigger
delete_trigger = _triggers.delete_trigger
has_handlers = _triggers.has_handlers
has_trigger = _triggers.has_trigger
is_blocked = _triggers.is_blocked
is_trigger_blocked = _triggers.is_trigger_blocked
manual_block = _triggers.manual_block
manual_release = _triggers.manual_release
remove_handler = _triggers.remove_handler
profile_trigger = _triggers.profile_trigger
release = _triggers.release
remove_handler = _triggers.remove_handler
trigger_handlers = _triggers.trigger_handlers
trigger_names = _triggers.trigger_names

# TODO:
# def add_handler(trigger_name: str, trigger_handler: Callable, object: Optional[Any] = None):
#    global _triggers
#    if object is None:
#        return _triggers.add_handler(trigger_name, trigger_handler)
#    else:
#        _handlers_by_object[object][trigger_name].add(
#            _triggers.add_handler(trigger_name, trigger_handler)
#        )

# TODO:
# def remove_handler_for_trigger(trigger_name, object):
#    _handlers_by_object[object][trigger_name]
#
# def remove_all_handlers(object):
#    for handler in _handlers_by_object[object]:
#        _triggers.remove_handler(handler)
#    del _handlers_by_object[object]
