from chimerax.core.triggerset import TriggerSet
from typing import Any, Callable, Optional

(ADDED, DELETED, EDITED) = manager_triggers = ("scenes added", "scenes deleted", "scenes edited")
(SCENE_SELECTED, SCENE_HIGHLIGHTED) = tool_triggers = ("scene selected", "scene highlighted")

"""
These triggers are all desiged to be triggered from the scene manager

ADDED: Trigger name for added scenes.
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
