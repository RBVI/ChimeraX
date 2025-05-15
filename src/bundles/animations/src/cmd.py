"""
This module defines commands for managing animations in ChimeraX.

Functions:
    register_command(command_name, logger): Register a command with the given name and logger.
    keyframe(session, action, keyframe_name, time): Edit actions on a keyframe in the animation StateManager.
    timeline(session): List all keyframes in the animation StateManager in the log.
    play(session, start_time, reverse): Play the animation from the Animation state manager.
    stop(session): Pause the animation that is playing.
    preview(session, time): Preview the animation at a specific time.
    set_length(session, length): Set the length of the animation.
    record(session, r_directory, r_pattern, r_format, r_size, r_supersample, r_transparent_background, r_limit,
        e_output, e_format, e_quality, e_qscale, e_bitrate, e_round_trip,
        e_reset_mode, e_wait, e_verbose): Record the animation using the movie bundle.
    stop_recording(session): Stop recording the animation and encode the movie.
    insert_time(session, target_time, amount_for_insertion): Insert time into the animation.
    remove_time(session, target_time, amount_for_removal): Remove time from the animation.
    clear(session): Clear all keyframes from the animation.
"""

from chimerax.core.commands import run, CmdDesc, register, ListOf, StringArg, FloatArg, BoolArg, IntArg, \
    SaveFileNameArg, EnumOf
from .animation import Animation
from chimerax.movie.movie import RESET_CLEAR, RESET_KEEP, RESET_NONE
from chimerax.movie.formats import formats, qualities, image_formats


def register_command(command_name, logger):
    """
    Register a command with the given name and logger.

    Args:
        command_name (str): The name of the command to register.
        logger: The logger to use for logging messages.
    """
    if command_name == "animations keyframe":
        func = keyframe
        desc = keyframe_desc
    elif command_name == "animations timeline":
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
    elif command_name == "animations insertTime":
        func = insert_time
        desc = insert_time_desc
    elif command_name == "animations removeTime":
        func = remove_time
        desc = remove_time_desc
    elif command_name == "animations clear":
        func = clear
        desc = clear_desc
    else:
        raise ValueError("trying to register unknown command: %s" % command_name)
    register(command_name, desc, func)


def keyframe(session, action: str, keyframe_name: str, time: int | float | None = None):
    """
    Edit actions on a keyframe in the animation StateManager. Add, edit, or delete keyframes.

    Args:
        session: The current session.
        action (str): The action to take on the keyframe. Options are add, edit, delete.
        keyframe_name (str): Name of the keyframe for the action to be applied to. Also be used for the scene name.
        time (int | float, optional): The time in seconds for the keyframe.
    """

    animation_mgr: Animation = session.get_state_manager("animations")

    if action == "add":
        if time is not None and not isinstance(time, (int, float)):
            session.logger.warning("Time must be an integer or float")
            return
        if animation_mgr.keyframe_exists(keyframe_name):
            session.logger.warning(f"Keyframe {keyframe_name} already exists")
            return
        run(session, f"scenes scene {keyframe_name}", log=False)
        animation_mgr.add_keyframe(keyframe_name, time)
    elif action == "edit":
        if not animation_mgr.keyframe_exists(keyframe_name):
            session.logger.warning(f"Keyframe {keyframe_name} does not exist")
            return
        if not isinstance(time, (int, float)):
            session.logger.warning("Time must be an integer or float")
            return
        animation_mgr.edit_keyframe_time(keyframe_name, time)
    elif action == "delete":
        if not animation_mgr.keyframe_exists(keyframe_name):
            session.logger.warning(f"Keyframe {keyframe_name} does not exist")
            return
        animation_mgr.delete_keyframe(keyframe_name)
    else:
        session.logger.warning(f"Action {action} not recognized. Options are add, edit, delete.")


keyframe_desc = CmdDesc(
    required=[
        ("action", StringArg),
        ("keyframe_name", StringArg),
    ],
    keyword=[
        ("time", FloatArg)
    ],
    synopsis="Create a keyframe in the animation StateManager."
)


def timeline(session):
    """
    List all keyframes in the animation StateManager in the log.

    Args:
        session: The current session.
    """
    animation_mgr: Animation = session.get_state_manager("animations")
    keyframes = animation_mgr.list_keyframes()
    for keyframe in keyframes:
        print(keyframe)


timeline_desc = CmdDesc(
    synopsis="List all keyframes in the animation StateManager."
)


def play(session, start_time=0, reverse=False):
    """
    Play the animation in the Animations state manager.

    Args:
        session: The current session.
        start_time (int | float, optional): Time to start playing the animation in seconds.
        reverse (bool, optional): True, play the animation in reverse. False, play the animation forward.
    """
    animation_mgr: Animation = session.get_state_manager("animations")
    if animation_mgr.get_num_keyframes() < 1:
        session.logger.warning("Need at least 1 keyframes to play the animation.")
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
    """
    Pause the animation in the StateManager. Only pauses if the animation is playing, and not if it is recording.

    Args:
        session: The current session.
    """
    animation_mgr: Animation = session.get_state_manager("animations")
    animation_mgr.stop_playing()


stop_desc = CmdDesc(
    synopsis="Stop the animation playing."
)


def preview(session, time: int | float):
    """
    Preview the animation at a specific time.

    Args:
        session: The current session.
        time (int | float): The time in seconds to preview the animation.
    """
    animation_mgr: Animation = session.get_state_manager("animations")
    if not isinstance(time, (int, float)):
        session.logger.warning("Time must be an integer or float")
        return
    if animation_mgr.get_num_keyframes() < 1:
        session.logger.warning("Need at least 1 keyframes to preview the animation.")
        return
    if not animation_mgr.time_in_range(time):
        session.logger.warning(f"Time must be between 0 and {animation_mgr.get_time_length()}")
        return
    animation_mgr.preview(time)


preview_desc = CmdDesc(
    required=[
        ("time", FloatArg)
    ],
    synopsis="Preview the animation at a specific time."
)


def set_length(session, length: int | float):
    """
    Set the length of the animation.

    Args:
        session: The current session.
        length (int | float): New length of the animation in seconds.
    """
    animation_mgr: Animation = session.get_state_manager("animations")
    animation_mgr.set_length(length)


set_length_desc = CmdDesc(
    required=[
        ("length", FloatArg)
    ],
    synopsis="Set the length of the animation."
)


def record(session, r_directory=None, r_pattern=None, r_format=None,
           r_size=None, r_supersample=1, r_transparent_background=False,
           r_limit=90000, e_output=None, e_format=None, e_quality=None, e_qscale=None, e_bitrate=None,
           e_round_trip=False, e_reset_mode=RESET_CLEAR, e_wait=False, e_verbose=False):
    """
    Record the animation using the movie bundle. The params are mirrored of what the movie record and encode command
    expects and get directly passed to the movie command inside the Animation class.

    Args:
        session: The current session.
        r_directory (str, optional): Directory to save the recording.
        r_pattern (str, optional): Pattern for the recording file names.
        r_format (str, optional): Format of the recording.
        r_size (str, optional): Size of the recording.
        r_supersample (int, optional): Supersample factor for the recording.
        r_transparent_background (bool, optional): Whether to use a transparent background.
        r_limit (int, optional): Frame limit for the recording.
        e_output (str, optional): Output file for the encoded movie.
        e_format (str, optional): Format of the encoded movie.
        e_quality (str, optional): Quality of the encoded movie.
        e_qscale (int, optional): Qscale for the encoded movie.
        e_bitrate (int, optional): Bitrate for the encoded movie.
        e_round_trip (bool, optional): Whether to round trip the movie.
        e_reset_mode (str, optional): Reset mode for the movie.
        e_wait (bool, optional): Whether to wait for the encoding to finish.
        e_verbose (bool, optional): Whether to enable verbose logging.
    """
    animation_mgr: Animation = session.get_state_manager("animations")
    if animation_mgr.get_num_keyframes() < 1:
        session.logger.warning("Need at least 1 keyframes to record the animation.")
        return

    from chimerax.movie import formats
    suffixes = set(fmt['suffix'] for fmt in formats.formats.values())
    if not e_output:
        session.logger.warning("Output file must be specified")
        return
    for output in e_output:
        if output and not any(output.endswith(suffix) for suffix in suffixes):
            session.logger.warning(f"Output file must have one of the following suffixes: {', '.join(suffixes)}")
            return


    record_params = {
        'directory': r_directory,
        'pattern': r_pattern,
        'format': r_format,
        'size': r_size,
        'supersample': r_supersample,
        'transparent_background': r_transparent_background,
        'limit': r_limit
    }

    # Framerate is omitted from the movie encoding param list because the animation manager will handle it.
    encode_params = {
        'output': e_output,
        'format': e_format,
        'quality': e_quality,
        'qscale': e_qscale,
        'bitrate': e_bitrate,
        'round_trip': e_round_trip,
        'reset_mode': e_reset_mode,
        'wait': e_wait,
        'verbose': e_verbose
    }

    animation_mgr.record(record_data=record_params, encode_data=encode_params)


ifmts = image_formats
fmts = tuple(formats.keys())
reset_modes = (RESET_CLEAR, RESET_KEEP, RESET_NONE)
record_desc = CmdDesc(
    optional=[('e_output', ListOf(SaveFileNameArg))],
    keyword=[('r_directory', SaveFileNameArg),
             ('r_pattern', StringArg),
             ('r_format', EnumOf(fmts)),
             ('r_size', ListOf(IntArg)),
             ('r_supersample', IntArg),
             ('r_transparent_background', BoolArg),
             ('r_limit', IntArg),
             ('e_output', ListOf),
             ('e_format', EnumOf(fmts)),
             ('e_quality', EnumOf(qualities)),
             ('e_qscale', IntArg),
             ('e_bitrate', FloatArg),
             ('e_reset_mode', EnumOf(reset_modes)),
             ('e_round_trip', BoolArg),
             ('e_wait', BoolArg),
             ('e_verbose', BoolArg),
             ],
    synopsis="Record the animation."
)


def stop_recording(session):
    """
    Stop recording the animation.

    Args:
        session: The current session.
    """
    animation_mgr: Animation = session.get_state_manager("animations")
    animation_mgr.stop_playing(stop_recording=True)


stop_recording_desc = CmdDesc(
    synopsis="Stop the recording of the animation."
)


def insert_time(session, target_time: int | float, time: int | float):
    """
    Insert time into the animation. All keyframes after the target time will be moved by the amount for insertion.

    Args:
        session: The current session.
        target_time (int | float): Time at which to insert the time in seconds.
        time (int | float): Amount of time to insert in seconds.
    """
    animation_mgr: Animation = session.get_state_manager("animations")
    animation_mgr.insert_time(target_time, time)


insert_time_desc = CmdDesc(
    required=[
        ("target_time", FloatArg),
        ("time", FloatArg)
    ],
    synopsis="Insert a segment of time at a target point on the timeline. Shift keyframes accordingly."
)


def remove_time(session, target_time: int | float, amount_for_removal: int | float):
    """
    Remove time from the animation. All keyframes after the target time will be moved closer to the target time.

    Args:
        session: The current session.
        target_time (int | float): Time at which to remove the time in seconds.
        amount_for_removal (int | float): Amount of time to remove in seconds.
    """
    animation_mgr: Animation = session.get_state_manager("animations")
    animation_mgr.remove_time(target_time, amount_for_removal)


remove_time_desc = CmdDesc(
    required=[
        ("target_time", FloatArg),
        ("amount_for_removal", FloatArg)
    ],
    synopsis="Remove a segment of time from the timeline. Shift keyframes accordingly."
)


def clear(session):
    """
    Clear all keyframes from the animation.

    Args:
        session: The current session.
    """
    animation_mgr: Animation = session.get_state_manager("animations")
    animation_mgr.delete_all_keyframes()


clear_desc = CmdDesc(
    synopsis="Remove all keyframes from the animations StateManager."
)
