# vi: set expandtab shiftwidth=4 softtabstop=4:


def set(session, bg_color=None, silhouettes=None, subdivision=None):
    '''Set global parameters.  With no options reports the current settings.

    Parameters
    ----------
    bg_color : Color
        Set the graphics window background color.
    silhouettes : bool
        Enable or disable silhouette edges (black lines drawn at boundaries of objects
        where the depth of the scene jumps.
    subdivision : float
        Controls the rendering quality of spheres and cylinders for drawing atoms and bonds.
        Default value is 1, higher values give smoother spheres and cylinders.
    '''
    had_arg = False
    view = session.main_view
    if bg_color is not None:
        had_arg = True
        view.background_color = bg_color.rgba
        view.redraw_needed = True
    if silhouettes is not None:
        had_arg = True
        view.silhouettes = silhouettes
        view.redraw_needed = True
    if subdivision is not None:
        had_arg = True
        from .. import atomic
        for m in atomic.all_atomic_structures(session):
            m.set_subdivision(subdivision)

    if not had_arg:
        msg = '\n'.join(('Current settings:',
                         '  bg_color: ' + str(view.background_color),
                         '  silhouettes: ' + str(view.silhouettes),
                         '  subdivision: ' + str(session.atomic_level_of_detail.quality)))
        session.logger.info(msg)


def register_command(session):
    from . import CmdDesc, register, ColorArg, BoolArg, FloatArg
    desc = CmdDesc(
        keyword=[('bg_color', ColorArg),
                 ('silhouettes', BoolArg),
                 ('subdivision', FloatArg)],
        synopsis="set preferences"
    )
    register('set', desc, set)
