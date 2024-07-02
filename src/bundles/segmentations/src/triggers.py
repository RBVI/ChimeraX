# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2024 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
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

AXIAL_PLANE_VIEWER_ENTER: Activated when the mouse enters a PlaneViewer
                          displaying an axial slice

CORONAL_PLANE_VIEWER_ENTER: Activated when the mouse enters a PlaneViewer
                            displaying a coronal slice

SAGITTAL_PLANE_VIEWER_ENTER: Activated when the mouse enters a PlaneViewer
                             displaying a sagittal slice

AXIAL_PLANE_VIEWER_LEAVE: Activated when the mouse leaves a PlaneViewer
                          displaying an axial slice

CORONAL_PLANE_VIEWER_LEAVE: Activated when the mouse leaves a PlaneViewer
                            displaying a coronal slice

SAGITTAL_PLANE_VIEWER_LEAVE: Activated when the mouse leaves a PlaneViewer
                             displaying a sagittal slice

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
"""

# from collections import defaultdict
from typing import Any, Callable, Optional

from chimerax.core.triggerset import TriggerSet

from chimerax.segmentations.types import Axis

SEGMENTATION_ADDED = "segmentation added"
SEGMENTATION_REMOVED = "segmentation removed"
SEGMENTATION_MODIFIED = "segmentation modified"

REFERENCE_MODEL_CHANGED = "reference model changed"
ACTIVE_SEGMENTATION_CHANGED = "active segmentation changed"

AXIAL_CURSOR_MOVED = "axial cursor moved"
CORONAL_CURSOR_MOVED = "coronal cursor moved"
SAGITTAL_CURSOR_MOVED = "sagittal cursor moved"
SPHERE_CURSOR_MOVED = "sphere cursor moved"

AXIAL_CURSOR_RESIZED = "axial cursor resized"
CORONAL_CURSOR_RESIZED = "coronal cursor resized"
SAGITTAL_CURSOR_RESIZED = "sagittal cursor resized"
SPHERE_CURSOR_RESIZED = "sphere cursor resized"

AXIAL_PLANE_VIEWER_ENTER = "axial plane viewer enter"
CORONAL_PLANE_VIEWER_ENTER = "coronal plane viewer enter"
SAGITTAL_PLANE_VIEWER_ENTER = "sagittal plane viewer enter"

AXIAL_PLANE_VIEWER_LEAVE = "axial plane viewer leave"
CORONAL_PLANE_VIEWER_LEAVE = "coronal plane viewer leave"
SAGITTAL_PLANE_VIEWER_LEAVE = "sagittal plane viewer leave"

VIEW_LAYOUT_CHANGED = "view layout changed"
GUIDELINES_VISIBILITY_CHANGED = "guidelines visibility changed"


ENTER_EVENTS = {
    Axis.AXIAL: AXIAL_PLANE_VIEWER_ENTER,
    Axis.CORONAL: CORONAL_PLANE_VIEWER_ENTER,
    Axis.SAGITTAL: SAGITTAL_PLANE_VIEWER_ENTER,
}

LEAVE_EVENTS = {
    Axis.AXIAL: AXIAL_PLANE_VIEWER_LEAVE,
    Axis.CORONAL: CORONAL_PLANE_VIEWER_LEAVE,
    Axis.SAGITTAL: SAGITTAL_PLANE_VIEWER_LEAVE,
}

_triggers = TriggerSet()

_triggers.add_trigger(SEGMENTATION_REMOVED)
_triggers.add_trigger(SEGMENTATION_ADDED)
_triggers.add_trigger(SEGMENTATION_MODIFIED)

_triggers.add_trigger(ACTIVE_SEGMENTATION_CHANGED)
_triggers.add_trigger(REFERENCE_MODEL_CHANGED)

_triggers.add_trigger(AXIAL_CURSOR_MOVED)
_triggers.add_trigger(CORONAL_CURSOR_MOVED)
_triggers.add_trigger(SAGITTAL_CURSOR_MOVED)
_triggers.add_trigger(SPHERE_CURSOR_MOVED)

_triggers.add_trigger(AXIAL_CURSOR_RESIZED)
_triggers.add_trigger(CORONAL_CURSOR_RESIZED)
_triggers.add_trigger(SAGITTAL_CURSOR_RESIZED)
_triggers.add_trigger(SPHERE_CURSOR_RESIZED)

_triggers.add_trigger(AXIAL_PLANE_VIEWER_ENTER)
_triggers.add_trigger(CORONAL_PLANE_VIEWER_ENTER)
_triggers.add_trigger(SAGITTAL_PLANE_VIEWER_ENTER)

_triggers.add_trigger(AXIAL_PLANE_VIEWER_LEAVE)
_triggers.add_trigger(CORONAL_PLANE_VIEWER_LEAVE)
_triggers.add_trigger(SAGITTAL_PLANE_VIEWER_LEAVE)


_triggers.add_trigger(VIEW_LAYOUT_CHANGED)
_triggers.add_trigger(GUIDELINES_VISIBILITY_CHANGED)

# TODO:
# _handlers_by_object = defaultdict(defaultdict(set).copy)


def activate_trigger(
    trigger_name: str, data: Optional[Any] = None, absent_okay: bool = False
) -> None:
    """
    Expected data by trigger:

    SEGMENTATION_REMOVED: the segmentation to be removed
    SEGMENTATION_ADDED: the segmentation about to be added
    SEGMENTATION_MODIFIED: the segmentation that has been modified

    ACTIVE_SEGMENTATION_CHANGED: the new active segmentation
    REFERENCE_MODEL_CHANGED: the new reference model (a volume)

    *_CURSOR_MOVED: the new coordinates of the axial cursor
    *_CURSOR_RESIZED: tuple(type of change[grow,shrink], resize step size(px))

    *_PLANE_VIEWER_ENTER: None
    *_PLANE_VIEWER_LEAVE: None

    VIEW_LAYOUT_CHANGED: the name of the new view layout
    GUIDELINES_TOGGLED: bool representing whether guidelines are on or off
    """
    global _triggers
    _triggers.activate_trigger(trigger_name, data)


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
