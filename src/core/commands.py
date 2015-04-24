# vi: set expandtab shiftwidth=4 softtabstop=4:
"""
commands -- Default set of commands
===================================

This module implements a default set of cli commands.
After importing this module, :py:func:`register`
must be called to get the commands recognized by the command line interface
(:py:mod:`chimera2.cli`).
"""

from . import atomspec
from . import cli
# from graphics.cameramode import CameraModeArg


def pwd(session):
    import os
    session.logger.info('current working directory: %s' % os.getcwd())
_pwd_desc = cli.CmdDesc()


def exit(session):
    session.ui.quit()
_exit_desc = cli.CmdDesc()


def stop(session, ignore=None):
    raise cli.UserError('use "exit" or "quit" instead of "stop"')
_stop_desc = cli.CmdDesc(optional=[('ignore', cli.RestOfLine)])


def echo(session, text=''):
    session.logger.info(text)
_echo_desc = cli.CmdDesc(optional=[('text', cli.RestOfLine)])


def open(session, filename, id=None, as_=None):
    try:
        return session.models.open(filename, id=id, as_=as_)
    except OSError as e:
        raise cli.UserError(e)
_open_desc = cli.CmdDesc(required=[('filename', cli.StringArg)],
                         keyword=[('id', cli.ModelIdArg),
                                  ('as', cli.StringArg)])


def export(session, filename, **kw):
    try:
        from . import io
        return io.export(session, filename, **kw)
    except OSError as e:
        raise cli.UserError(e)
_export_desc = cli.CmdDesc(required=[('filename', cli.StringArg)])


def close(session, model_ids):
    m = session.models
    try:
        mlist = sum((m.list(model_id) for model_id in model_ids), [])
    except ValueError as e:
        raise cli.UserError(e)
    m.close(mlist)
_close_desc = cli.CmdDesc(required=[('model_ids', cli.ListOf(cli.ModelIdArg))])


def list(session):
    models = session.models.list()
    if len(models) == 0:
        session.logger.status("No open models.")
        return

    def id_str(id):
        if isinstance(id, int):
            return str(id)
        return '.'.join(str(x) for x in id)
    ids = [m.id for m in models]
    ids.sort()
    info = "Open models: "
    if len(models) > 1:
        info += ", ".join(id_str(id) for id in ids[:-1]) + " and"
    info += " %s" % id_str(ids[-1])
    session.logger.info(info)
_list_desc = cli.CmdDesc()


def help(session, command_name=None):
    from . import cli
    status = session.logger.status
    info = session.logger.info
    if command_name is None:
        info("Use 'help <command>' to learn more about a command.")
        cmds = cli.registered_commands()
        cmds.sort()
        if len(cmds) == 0:
            pass
        elif len(cmds) == 1:
            info("The following command is available: %s" % cmds[0])
        else:
            info("The following commands are available: %s, and %s"
                 % (', '.join(cmds[:-1]), cmds[-1]))
        return
    try:
        usage = cli.usage(command_name)
    except ValueError as e:
        status(str(e))
        return
    status(usage)
    info(cli.html_usage(command_name), is_html=True)
_help_desc = cli.CmdDesc(optional=[('command_name', cli.StringArg)])


def display(session, spec=None):
    if spec is None:
        spec = atomspec.everything(session)
    results = spec.evaluate(session)
    results.atoms.displays = True
    for m in results.models:
        m.update_graphics()
_display_desc = cli.CmdDesc(optional=[("spec", atomspec.AtomSpecArg)])


def undisplay(session, spec=None):
    if spec is None:
        spec = atomspec.everything(session)
    results = spec.evaluate(session)
    results.atoms.displays = False
    for m in results.models:
        m.update_graphics()
_undisplay_desc = cli.CmdDesc(optional=[("spec", atomspec.AtomSpecArg)])


def window(session):
    session.main_view.view_all()
_window_desc = cli.CmdDesc()


def camera(session, mode=None, field_of_view=None, eye_separation=None,
           screen_width=None, depth_scale=None):
    view = session.main_view
    cam = session.main_view.camera
    has_arg = False
    if mode is not None:
        has_arg = True
        # TODO
    if field_of_view is not None:
        has_arg = True
        cam.field_of_view = field_of_view
        cam.redraw_needed = True
    if eye_separation is not None or screen_width is not None:
        has_arg = True
        if eye_separation is None or screen_width is None:
            raise cli.UserError("Must specifiy both eye-separation and"
                                " screen-width -- only ratio is used")
        cam.eye_separation_pixels = (eye_separation / screen_width) * \
            view.screen().size().width()
        cam.redraw_needed = True
    if depth_scale is not None:
        has_arg = True
        cam.eye_separation_pixels *= depth_scale
        cam.eye_separation_scene *= depth_scale
        cam.redraw_needed = True
    if not has_arg:
        msg = (
            'Camera parameters:\n' +
            '    position: %.5g %.5g %.5g\n' % tuple(cam.position.origin()) +
            '    view direction: %.6f %.6f %.6f\n' %
            tuple(cam.view_direction()) +
            '    field of view: %.5g degrees\n' % cam.field_of_view +
            '    mode: %s\n' % cam.mode.name()
        )
        session.logger.info(msg)
        msg = (cam.mode.name() +
            ', %.5g degree field of view' % cam.field_of_view)
        session.logger.status(msg)

_camera_desc = cli.CmdDesc(optional=[
    # ('mode', CameraModeArg),
    ('field_of_view', cli.FloatArg),
    ('eye_separation', cli.FloatArg),
    ('screen_width', cli.FloatArg),
    ('depth_scale', cli.FloatArg),
])

def save(session, filename, width = None, height = None, format = None, supersample = None):
    from os.path import splitext
    e = splitext(filename)[1].lower()
    from . import session as ses
    if e[1:] in image_file_suffixes:
        save_image(session, filename, width, height, format, supersample)
    elif e == ses.SUFFIX:
        ses.save(session, filename)
    else:
        suffixes = image_file_suffixes + (ses.SUFFIX[1:],)
        raise cli.UserError('Unrecognized file suffix "%s", require one of %s'
                            % (e, ','.join(suffixes)))

_save_desc = cli.CmdDesc(
    required = [('filename', cli.StringArg),],
    keyword=[('width', cli.IntArg),
             ('height', cli.IntArg),
             ('supersample', cli.IntArg),
             ('quality', cli.IntArg),
             ('format', cli.StringArg),])

# Table mapping file suffix to Pillow image format.
image_formats = {
    'png': 'PNG',
    'jpg': 'JPEG',
    'tif': 'TIFF',
    'gif': 'GIF',
    'ppm': 'PPM',
    'bmp': 'BMP',
}
image_file_suffixes = tuple(image_formats.keys())
def save_image(session, path, format = None, width = None, height = None, supersample = None, quality = 95):
    '''
    Save an image of the current graphics window contents.
    '''
    from os.path import expanduser, dirname, exists, splitext
    path = expanduser(path)         # Tilde expansion
    dir = dirname(path)
    if not exists(dir):
        raise cli.UserError('Directory "%s" does not exist' % dir)

    if format is None:
        suffix = splitext(path)[1][1:].lower()
        if not suffix in image_file_suffixes:
            raise cli.UserError('Unrecognized image file suffix "%s"' % format)
        format = image_formats[suffix]

    view = session.main_view
    i = view.image(width, height, supersample = supersample)
    i.save(path, format, quality = quality)

def set(session, bgColor = None, silhouette = None):
    view = session.main_view
    if not bgColor is None:
        view.background_color = bgColor.rgba
        view.redraw_needed = True
    if not silhouette is None:
        view.silhouettes = silhouette
        view.redraw_needed = True

from .color import ColorArg
_set_desc = cli.CmdDesc(
    keyword = [('bgColor', ColorArg),
               ('silhouette', cli.BoolArg),
           ]
)

def register(session):
    """Register common cli commands"""
    import sys
    cli.register('exit', _exit_desc, exit)
    cli.alias(session, "quit", "exit $*")
    cli.register('open', _open_desc, open)
    cli.register('close', _close_desc, close)
    cli.register('export', _export_desc, export)
    cli.register('list', _list_desc, list)
    cli.register('stop', _stop_desc, stop)
    cli.register('echo', _echo_desc, echo)
    cli.register('pwd', _pwd_desc, pwd)
    cli.register('window', _window_desc, window)
    cli.register('help', _help_desc, help)
    cli.register('display', _display_desc, display)
    cli.register('~display', _undisplay_desc, undisplay)
    cli.register('camera', _camera_desc, camera)
    cli.register('save', _save_desc, save)
    cli.register('set', _set_desc, set)
    from . import molsurf
    molsurf.register_surface_command()
    molsurf.register_sasa_command()
    molsurf.register_buriedarea_command()
    from . import structure
    structure.register_molecule_commands()
    from . import lightcmd
    lightcmd.register_lighting_command()
    lightcmd.register_material_command()
    from . import map
    map.register_volume_command()
    from .map import series
    series.register_vseries_command()
    from . import color
    color.register_commands()
    from .devices import oculus
    oculus.register_oculus_command()
    from .devices import spacenavigator
    spacenavigator.register_snav_command()
    from . import shortcuts
    shortcuts.register_shortcut_command()

    # def lighting_cmds():
    #     import .lighting.cmd as cmd
    #     cmd.register()
    # cli.delay_registration('lighting', lighting_cmds)
