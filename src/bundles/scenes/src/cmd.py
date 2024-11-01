from chimerax.core.commands import register, CmdDesc, StringArg, FloatArg


def register_command(command_name, logger):
    if command_name == "scenes save":
        func = save_scene
        desc = save_scene_desc
    elif command_name == "scenes delete":
        func = delete_scene
        desc = delete_scene_desc
    elif command_name == "scenes restore":
        func = restore_scene
        desc = restore_scene_desc
    elif command_name == "scenes interpolate":
        func = interpolate_scenes
        desc = interpolate_scenes_desc
    elif command_name == "scenes dInterpolate":
        func = dynamic_interpolate_scenes
        desc = dynamic_interpolate_scenes_desc
    elif command_name == "scenes list":
        func = list_scenes
        desc = list_scenes_desc
    else:
        raise ValueError("trying to register unknown command: %s" % command_name)
    register(command_name, desc, func)


def save_scene(session, scene_name):
    """Save the current scene as 'scene_name'."""
    session.scenes.save_scene(scene_name)


save_scene_desc = CmdDesc(
    required=[("scene_name", StringArg)],
    synopsis="Save the current scene as 'scene_name'."
)


def delete_scene(session, scene_name: str):
    """Delete the current scene as 'scene_name'."""
    session.scenes.delete_scene(scene_name)


delete_scene_desc = CmdDesc(
    required=[("scene_name", StringArg)],
    synopsis="Delete scene 'scene_name'."
)


def restore_scene(session, scene_name):
    """Restore the scene named 'scene_name'."""
    session.scenes.restore_scene(scene_name)


restore_scene_desc = CmdDesc(
    required=[("scene_name", StringArg)],
    synopsis="Restore the scene named 'scene_name'."
)


def interpolate_scenes(session, scene_name1, scene_name2, fraction):
    """Interpolate between two scenes."""
    if fraction < 0.0 or fraction > 1.0:
        print("Fraction must be between 0.0 and 1.0")
        return
    session.scenes.interpolate_scenes(scene_name1, scene_name2, fraction)


interpolate_scenes_desc = CmdDesc(
    required=[
        ("scene_name1", StringArg),
        ("scene_name2", StringArg),
        ("fraction", FloatArg)
    ],
    synopsis="Interpolate between two scenes."
)


def dynamic_interpolate_scenes(session, scene_name1, scene_name2):
    """Interpolate between two scenes using a fraction but make the animation take 5 seconds and split it into
    60fps"""
    from chimerax.core.commands import run
    import time
    from chimerax.core.commands.motion import CallForNFrames

    def frame_func(session, f):
        fraction = f / 300
        session.scenes.interpolate_scenes(scene_name1, scene_name2, fraction)

    CallForNFrames(frame_func, 300, session)


dynamic_interpolate_scenes_desc = CmdDesc(
    required=[
        ("scene_name1", StringArg),
        ("scene_name2", StringArg)
    ],
    synopsis="Interpolate between two scenes dynamically."
)


def list_scenes(session):
    """List all saved scenes."""
    for scene_name in session.scenes.scenes.keys():
        print(scene_name)


list_scenes_desc = CmdDesc(
    synopsis="List all saved scenes."
)