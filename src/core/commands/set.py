# vim: set expandtab shiftwidth=4 softtabstop=4:


def set(session, bg_color=None,
        silhouettes=None, silhouette_width=None, silhouette_color=None, silhouette_depth_jump=None,
        subdivision=None):
    '''Set global parameters.  With no options reports the current settings.

    Parameters
    ----------
    bg_color : Color
        Set the graphics window background color.
    silhouettes : bool
        Enable or disable silhouette edges (black lines drawn at boundaries of objects
        where the depth of the scene jumps.
    silhouette_width : float
        Width in pixels of silhouette edges. Minimum width is 1 pixel.
    silhouette_color : color
        Color of silhouette edges.
    silhouette_depth_jump : float
        Fraction of scene depth giving minimum depth change to draw a silhouette edge. Default 0.03.
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
    if silhouette_width is not None:
        had_arg = True
        view.silhouette_thickness = silhouette_width
        view.redraw_needed = True
    if silhouette_color is not None:
        had_arg = True
        view.silhouette_color = silhouette_color.rgba
        view.redraw_needed = True
    if silhouette_depth_jump is not None:
        had_arg = True
        view.silhouette_depth_jump = silhouette_depth_jump
        view.redraw_needed = True
    if subdivision is not None:
        had_arg = True
        from .. import atomic
        for m in atomic.all_atomic_structures(session):
            m.set_subdivision(subdivision)

    if not had_arg:
        msg = '\n'.join(('Current settings:',
                         '  Background color: %d,%d,%d' % tuple(100*r for r in view.background_color[:3]),
                         '  Silhouettes: ' + str(view.silhouettes),
                         '  Silhouette width: %.3g' % view.silhouette_thickness,
                         '  Silhouette color: %d,%d,%d' % tuple(100*r for r in view.silhouette_color[:3]),
                         '  Silhouette depth jump: %.3g' % view.silhouette_depth_jump,
                         '  Subdivision: %.3g'  % session.atomic_level_of_detail.quality))
        session.logger.info(msg)


def register_command(session):
    from . import CmdDesc, register, ColorArg, BoolArg, FloatArg
    desc = CmdDesc(
        keyword=[('bg_color', ColorArg),
                 ('silhouettes', BoolArg),
                 ('silhouette_width', FloatArg),
                 ('silhouette_color', ColorArg),
                 ('silhouette_depth_jump', FloatArg),
                 ('subdivision', FloatArg)],
        synopsis="set preferences"
    )
    register('set', desc, set)
