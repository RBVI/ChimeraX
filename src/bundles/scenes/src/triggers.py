# === UCSF ChimeraX Copyright ===
# Copyright 2025 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# You can also
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

from chimerax.core.triggerset import TriggerSet
from typing import Any, Callable, Optional

(SAVED, DELETED, EDITED, RESTORED) = manager_triggers = ("scene saved", "scene deleted", "scene edited",
	"scene restored")
(SCENE_SELECTED, SCENE_HIGHLIGHTED) = tool_triggers = ("scene selected", "scene highlighted")

"""
These triggers are all designed to be triggered from the scene manager

SAVED: Trigger name for saved scenes.
DELETED: Trigger name for deleted scenes.
EDITED: Trigger name for edited scenes.
"""

"""
These triggers are designed to be used in the bundle's tool

SCENE_SELECTED: Trigger from the tool for when a scene needs to be restored.
SCENE_HIGHLIGHTED: Trigger name for a SceneItem widget that was highlighted.
"""

_triggers = TriggerSet()

for trigger in manager_triggers:
    _triggers.add_trigger(trigger)

for trigger in tool_triggers:
    _triggers.add_trigger(trigger)


def activate_trigger(trigger_name: str, data: Optional[Any] = None, absent_okay: bool = False) -> None:
    _triggers.activate_trigger(trigger_name, data)


def add_handler(trigger_name: str, func: Callable):
    # Returns _TriggerHandler instance
    return _triggers.add_handler(trigger_name, func)


def remove_handler(handler) -> None:
    _triggers.remove_handler(handler)
