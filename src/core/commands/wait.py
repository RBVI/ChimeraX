# vi: set expandtab shiftwidth=4 softtabstop=4:

def wait(session, frames=None):
    '''Wait before proceeding to the next command. Used in movie recording scripts.

    Parameters
    ----------
    frames : integer
       Wait until this many frames have been rendered before executing the next
       command in a command script.
    '''
    v = session.main_view
    if frames is None:
        from . import motion
        while motion.motion_in_progress(session):
            v.redraw_needed = True  # Trigger frame rendered callbacks to cause image capture.
            v.draw_new_frame()
    else:
        for f in range(frames):
            v.redraw_needed = True  # Trigger frame rendered callbacks to cause image capture.
            v.draw_new_frame()

def register_command(session):
    from . import cli
    desc = cli.CmdDesc(
        optional=[('frames', cli.PositiveIntArg)],
        synopsis='suspend command processing for a specified number of frames'
        ' or until finite motions have stopped ')
    cli.register('wait', desc, wait)
