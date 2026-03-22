from chimerax.core.triggerset import TriggerSet

from typing import Optional, Any, Callable

from enum import StrEnum

# Signals for once the animation manager has made an action
(MGR_KF_ADDED, MGR_KF_DELETED, MGR_KF_EDITED, MGR_LENGTH_CHANGED, MGR_PREVIEWED, MGR_FRAME_PLAYED, MGR_RECORDING_START,
 MGR_RECORDING_STOP) = manager_triggers = (
    "animations mgr keyframe added", "animations mgr keyframe deleted", "animations mgr keyframe edited",
    "animations mgr length changed", "animations mgr previewed", "animations mgr frame played",
    "animations mgr recording start", "animations mgr recording stop")

"""
All MGR_ prefix commands are triggered by the animations manager once an action has been completed.

MGR_KF_ADDED: Triggered when a keyframe is added to the animation manager. Data is reference to the
animation.Keyframe object that was added.

MGR_KF_DELETED: Triggered when a keyframe is deleted from the animation manager. Data is reference to the
animation.Keyframe object that was removed from the animation manager.

MGR_KF_EDITED: Triggered when a keyframe is edited in the animation manager. Data is reference to the
animation.Keyframe object that was edited.

MGR_LENGTH_CHANGED: Triggered when the length of the animation is changed. Data is the new length of the animation in
seconds. int/float.

MGR_PREVIEWED: Triggered when the animation manager previews a frame. Data is the time in seconds (int/float) that is
getting previewed.

MGR_FRAME_PLAYED: Triggered when the animation manager plays a frame. Data is the time in seconds (int/float) of the
frame that is being shown.
"""

# Signals for if the animation manager needs to make an action
(KF_ADD, KF_DELETE, KF_EDIT, LENGTH_CHANGE, PREVIEW, PLAY, RECORD, STOP_PLAYING, REMOVE_TIME,
 INSERT_TIME, STOP_RECORDING) = external_triggers = (
    "animations keyframe add", "animations keyframe delete", "animations keyframe edit", "animations length change",
    "animations preview", "animations play", "animations record", "animations stop playing", "animations remove time",
    "animations insert time", "animations stop recording")

"""
Non MGR_ prefix commands are triggered by external sources and are handled by the tool to make command calls to the
animation manager.

KF_ADD: Triggered when a keyframe needs to be added to the animation manager. Data is the time in seconds (int/float)
to add the keyframe at.

KF_DELETE: Triggered when a keyframe needs to be deleted from the animation manager. Data is the name (str) of the
keyframe that needs to be deleted.

KF_EDIT: Triggered when a keyframe needs to be edited in the animation manager. Data is a tuple (keyframe name, time)
(str, int/float) of the keyframe that needs to be edited and the new time in seconds.

LENGTH_CHANGE: Triggered when the length of the animation needs to be changed. Data is the new length of the
animation in seconds (int/float).

PREVIEW: Triggered when the animation manager needs to preview a frame. Data is the time in seconds (int/float) to
preview

PLAY: Triggered when the animation manager needs to play a frame. Data is a tuple (time, reverse) (int/float, bool).
The time is in seconds for when to start playing and reverse is a boolean for if the animation should play in reverse.
True for reverse, False for forward.

RECORD: Triggered when the animation manager needs to record the animation. Data is None.

STOP_PLAYING: Triggered when the animation manager needs to stop playing the animation. Data is None.

REMOVE_TIME: Triggered when the animation manager needs to remove a time segment from the animation. Data is
(target time, time to remove) in seconds (int/float, int/float).

STOP_RECORDING: Triggered when the animation manager needs to stop recording the animation. Data is None.
"""

class Trigger(StrEnum):
    MGR_KEYFRAME_ADDED = "animations mgr keyframe added"
    MGR_KEYFRAME_DELETED = "animations mgr keyframe deleted"
    MGR_KEYFRAME_EDITED = "animations mgr keyframe edited"
    MGR_LENGTH_CHANGED = "animations mgr length changed"
    MGR_PREVIEWED = "animations mgr previewed"
    MGR_FRAME_PLAYED = "animations mgr frame played"
    MGR_RECORDING_START = "animations mgr recording start"
    MGR_RECORDING_STOP = "animations mgr recording stop"


_triggers = TriggerSet()

for trigger in manager_triggers:
    _triggers.add_trigger(trigger)

for trigger in external_triggers:
    _triggers.add_trigger(trigger)


def activate_trigger(trigger_name: str, data: Optional[Any] = None, absent_okay: bool = False) -> None:
    _triggers.activate_trigger(trigger_name, data)


def add_handler(trigger_name: str, func: Callable):
    # Returns _TriggerHandler instance
    return _triggers.add_handler(trigger_name, func)


def remove_handler(handler) -> None:
    _triggers.remove_handler(handler)
