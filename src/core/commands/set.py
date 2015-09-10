# vi: set expandtab shiftwidth=4 softtabstop=4:

def set(session, bg_color=None, silhouettes=None):
    '''Set global parameters.  With no options reports the current settings.

    Parameters
    ----------
    bg_color : Color
        Set the graphics window background color.
    silhouettes : bool
        Enable or disable silhouette edges (black lines drawn at boundaries of objects
        where the depth of the scene jumps.
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
    if had_arg:
        return
    print('Current settings:\n'
          '  bg_color:', view.background_color, '\n'
          '  silhouettes:', view.silhouettes, '\n')

def register_command(session):
    from . import cli, color, ColorArg
    desc = cli.CmdDesc(
        keyword=[('bg_color', ColorArg), ('silhouettes', cli.BoolArg)],
        synopsis="set preferences"
        )
    cli.register('set', desc, set)

