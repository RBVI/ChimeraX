"""
Commands for managing animations in ChimeraX.

Commands dispatch to either the keyframe-based Animation or the scene-based
SceneAnimation state manager depending on the current animation_mode setting.
"""

from chimerax.core.commands import (
    CmdDesc, register, FloatArg, BoolArg, SaveFileNameArg,
)


def _get_mode(session):
    """Return the current animation mode ('scene' or 'keyframe')."""
    from .settings import get_settings
    return get_settings(session).animation_mode


def _get_scene_animation(session):
    """Return the session-registered SceneAnimation state manager."""
    return session.get_state_manager("scene animations")


def _get_keyframe_animation(session):
    """Return the session-registered keyframe Animation state manager."""
    return session.get_state_manager("animations")


# ---------------------------------------------------------------------------
# Command registration
# ---------------------------------------------------------------------------

def register_command(command_name, logger):
    if command_name == "animations timeline":
        func = timeline
        desc = timeline_desc
    elif command_name == "animations play":
        func = play
        desc = play_desc
    elif command_name == "animations stop":
        func = stop
        desc = stop_desc
    elif command_name == "animations preview":
        func = preview
        desc = preview_desc
    elif command_name == "animations setLength":
        func = set_length
        desc = set_length_desc
    elif command_name == "animations record":
        func = record
        desc = record_desc
    elif command_name == "animations stopRecording":
        func = stop_recording
        desc = stop_recording_desc
    elif command_name == "animations clear":
        func = clear
        desc = clear_desc
    else:
        raise ValueError("trying to register unknown command: %s" % command_name)
    register(command_name, desc, func)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def timeline(session):
    if _get_mode(session) == "scene":
        mgr = _get_scene_animation(session)
        if not mgr.scenes:
            session.logger.info("No scenes in the animation.")
            return
        session.logger.info(f"Animation duration: {mgr.duration:.2f}s")
        for time, scene_name, transition_data in mgr.scenes:
            ttype = transition_data.get("type", "linear") if transition_data else "linear"
            session.logger.info(f"  {time:.2f}s: {scene_name} ({ttype})")
    else:
        animation_mgr = _get_keyframe_animation(session)
        keyframes = animation_mgr.list_keyframes()
        for kf in keyframes:
            session.logger.info(kf)


timeline_desc = CmdDesc(
    synopsis="List all keyframes or scenes in the animation."
)


def play(session, start_time=0, reverse=False):
    if _get_mode(session) == "scene":
        mgr = _get_scene_animation(session)
        if not mgr.scenes and not mgr.action_segments:
            session.logger.warning("No scenes to play.")
            return
        mgr.play(start_time, reverse)
    else:
        animation_mgr = _get_keyframe_animation(session)
        if animation_mgr.get_num_keyframes() < 1:
            session.logger.warning("Need at least 1 keyframe to play the animation.")
            return
        animation_mgr.play(start_time, reverse)


play_desc = CmdDesc(
    keyword=[
        ("start_time", FloatArg),
        ("reverse", BoolArg)
    ],
    synopsis="Play the animation."
)


def stop(session):
    if _get_mode(session) == "scene":
        _get_scene_animation(session).stop_playing()
    else:
        _get_keyframe_animation(session).stop_playing()


stop_desc = CmdDesc(
    synopsis="Stop the animation playing."
)


def preview(session, time):
    if _get_mode(session) == "scene":
        mgr = _get_scene_animation(session)
        if not mgr.scenes and not mgr.action_segments:
            session.logger.warning("No scenes to preview.")
            return
        if time < 0 or time > mgr.duration:
            session.logger.warning(
                f"Time must be between 0 and {mgr.duration}"
            )
            return
        mgr.preview_at_time(time)
    else:
        animation_mgr = _get_keyframe_animation(session)
        if not isinstance(time, (int, float)):
            session.logger.warning("Time must be an integer or float")
            return
        if animation_mgr.get_num_keyframes() < 1:
            session.logger.warning(
                "Need at least 1 keyframe to preview the animation."
            )
            return
        if not animation_mgr.time_in_range(time):
            session.logger.warning(
                f"Time must be between 0 and {animation_mgr.get_time_length()}"
            )
            return
        animation_mgr.preview(time)


preview_desc = CmdDesc(
    required=[
        ("time", FloatArg)
    ],
    synopsis="Preview the animation at a specific time."
)


def set_length(session, length):
    if _get_mode(session) == "scene":
        _get_scene_animation(session).set_duration(length)
    else:
        _get_keyframe_animation(session).set_length(length)


set_length_desc = CmdDesc(
    required=[
        ("length", FloatArg)
    ],
    synopsis="Set the length of the animation."
)


RESOLUTION_CHOICES = (
    "graphics_display", "4k", "1080p", "720p", "480p",
)


def record(session, output=None, resolution=None):
    mgr = _get_scene_animation(session)
    if not mgr.scenes:
        session.logger.warning("No scenes to record.")
        return
    if not output:
        session.logger.warning("Output file must be specified")
        return
    # "graphics_display" means use the current window size, which
    # SceneAnimation.record treats as None.
    if resolution == "graphics_display":
        resolution = None
    mgr.record(output, resolution=resolution)


from chimerax.core.commands import EnumOf

record_desc = CmdDesc(
    required=[('output', SaveFileNameArg)],
    keyword=[('resolution', EnumOf(RESOLUTION_CHOICES))],
    synopsis="Record the animation."
)


def stop_recording(session):
    if _get_mode(session) == "scene":
        _get_scene_animation(session).stop_playing()
    else:
        _get_keyframe_animation(session).stop_playing(stop_recording=True)


stop_recording_desc = CmdDesc(
    synopsis="Stop the recording of the animation."
)


def clear(session):
    if _get_mode(session) == "scene":
        _get_scene_animation(session).clear_timeline()
    else:
        _get_keyframe_animation(session).delete_all_keyframes()


clear_desc = CmdDesc(
    synopsis="Remove all keyframes or scenes from the animation."
)
