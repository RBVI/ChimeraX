# vim: set expandtab shiftwidth=4 softtabstop=4:

def run_preset(session, name, mgr):
    """Run requested preset."""

    # A preset needs to call mgr.execute(preset_info) to
    # execute the preset, so that information about the
    # preset's contents can be properly logged.  The
    # 'preset_info' argument can either be an executable
    # Python function (that takes no arguments) or a
    # string containing one or more ChimeraX commands.
    # If there are multiple commands, the commands are typically
    # separated with '; ', though in unusual cases (perhaps
    # a very long series of commands) the commands could be
    # newline separated -- in the latter case the newline-
    # separated command string will be executed in individual
    # calls to chimerax.core.commands.run() whereas '; '-
    # separated commands will use only one call.
    #
    # Here we form a command string and use it with mgr.execute()
    base_cmd = "size stickRadius 0.07 ballScale 0.18"
    if name == "thin sticks":
        style_cmd = "style stick"
    elif name == "ball and stick":
        style_cmd = "style ball"
    else:
        raise ValueError("No preset named '%s'" % name)
    mgr.execute(base_cmd + '; ' + style_cmd)

