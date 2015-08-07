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
from .errors import UserError
# from graphics.cameramode import CameraModeArg


def pwd(session):
    import os
    session.logger.info('current working directory: %s' % os.getcwd())
_pwd_desc = cli.CmdDesc(synopsis='print current working directory')


def exit(session):
    session.ui.quit()
_exit_desc = cli.CmdDesc(synopsis='exit application')


def stop(session, ignore=None):
    raise UserError('use "exit" or "quit" instead of "stop"')
_stop_desc = cli.CmdDesc(optional=[('ignore', cli.RestOfLine)],
                         synopsis='DO NOT USE')


def echo(session, text=''):
    tokens = []
    while text:
        token, chars, rest = cli.next_token(text)
        tokens.append(token)
        m = cli._whitespace.match(rest)
        rest = rest[m.end():]
        text = rest
    text = ' '.join(tokens)
    session.logger.info(text)
_echo_desc = cli.CmdDesc(optional=[('text', cli.RestOfLine)],
                         synopsis='show text in log')


def open(session, filename, id=None, as_=None):
    try:
        return session.models.open(filename, id=id, as_=as_)
    except OSError as e:
        raise UserError(e)
_open_desc = cli.CmdDesc(required=[('filename', cli.StringArg)],
                         keyword=[('id', cli.ModelIdArg),
                                  ('as_a', cli.StringArg),
                                  ('label', cli.StringArg)],
                         synopsis='read and display data')


def export(session, filename, **kw):
    try:
        from . import io
        return io.export(session, filename, **kw)
    except OSError as e:
        raise UserError(e)
_export_desc = cli.CmdDesc(required=[('filename', cli.StringArg)],
                           synopsis='export data in format'
                           ' matching filename suffix')


def close(session, model_ids=None):
    m = session.models
    if model_ids is None:
        mlist = m.list()
    else:
        try:
            mlist = sum((m.list(model_id) for model_id in model_ids), [])
        except ValueError as e:
            raise UserError(e)
    m.close(mlist)
_close_desc = cli.CmdDesc(optional=[('model_ids', cli.ListOf(cli.ModelIdArg))],
                          synopsis='close models')


def delete(session, atoms):
    atoms.delete()
from .structure import AtomsArg
_delete_desc = cli.CmdDesc(required=[('atoms', AtomsArg)],
                           synopsis='delete atoms')


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
_list_desc = cli.CmdDesc(synopsis='list open model ids')


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
    elif command_name == 'all':
        info("Syntax for all commands.")
        cmds = cli.registered_commands()
        cmds.sort()
        for name in cmds:
            try:
                info(cli.html_usage(name), is_html=True)
            except:
                info('<b>%s</b> no documentation' % name, is_html=True)
        return

    try:
        usage = cli.usage(command_name)
    except ValueError as e:
        status(str(e))
        return
    if session.ui.is_gui:
        info(cli.html_usage(command_name), is_html=True)
    else:
        info(usage)
_help_desc = cli.CmdDesc(optional=[('command_name', cli.RestOfLine)],
                         synopsis='show command usage')


def display(session, spec=None):
    if spec is None:
        spec = atomspec.everything(session)
    results = spec.evaluate(session)
    results.atoms.displays = True
_display_desc = cli.CmdDesc(optional=[("spec", atomspec.AtomSpecArg)],
                            synopsis='display specified atoms')


def undisplay(session, spec=None):
    if spec is None:
        spec = atomspec.everything(session)
    results = spec.evaluate(session)
    results.atoms.displays = False
_undisplay_desc = cli.CmdDesc(optional=[("spec", atomspec.AtomSpecArg)],
                              synopsis='undisplay specified atoms')


def window(session):
    session.main_view.view_all()
_window_desc = cli.CmdDesc(synopsis='reset view so everything is visible in window')


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
            raise UserError("Must specifiy both eye-separation and"
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

_camera_desc = cli.CmdDesc(
    optional=[
        # ('mode', CameraModeArg),
        ('field_of_view', cli.FloatArg),
        ('eye_separation', cli.FloatArg),
        ('screen_width', cli.FloatArg),
        ('depth_scale', cli.FloatArg),
    ],
    synopsis='adjust camara parameters'
)


def save(session, filename, width=None, height=None, supersample=None, format=None):
    from os.path import splitext
    e = splitext(filename)[1].lower()
    from . import session as ses
    if e[1:] in image_file_suffixes:
        save_image(session, filename, format, width, height, supersample)
    elif e == ses.SESSION_SUFFIX:
        ses.save(session, filename)
    else:
        suffixes = image_file_suffixes + (ses.SESSION_SUFFIX[1:],)
        raise UserError('Unrecognized file suffix "%s", require one of %s'
                            % (e, ','.join(suffixes)))

_save_desc = cli.CmdDesc(
    required=[('filename', cli.StringArg), ],
    keyword=[
        ('width', cli.IntArg),
        ('height', cli.IntArg),
        ('supersample', cli.IntArg),
        ('quality', cli.IntArg),
        ('format', cli.StringArg),
    ],
    synopsis='save session or image'
)

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


def save_image(session, path, format=None, width=None, height=None,
               supersample=None, quality=95):
    '''
    Save an image of the current graphics window contents.
    '''
    from os.path import expanduser, dirname, exists, splitext
    path = expanduser(path)         # Tilde expansion
    dir = dirname(path)
    if dir and not exists(dir):
        raise UserError('Directory "%s" does not exist' % dir)

    if format is None:
        suffix = splitext(path)[1][1:].lower()
        if suffix not in image_file_suffixes:
            raise UserError('Unrecognized image file suffix "%s"' % format)
        format = image_formats[suffix]

    view = session.main_view
    i = view.image(width, height, supersample=supersample)
    i.save(path, format, quality=quality)


def ribbon(session, spec=None):
    if spec is None:
        spec = atomspec.everything(session)
    results = spec.evaluate(session)
    results.atoms.residues.ribbon_displays = True

_ribbon_desc = cli.CmdDesc(optional=[("spec", atomspec.AtomSpecArg)],
                           synopsis='display ribbon for specified residues')


def unribbon(session, spec=None):
    if spec is None:
        spec = atomspec.everything(session)
    results = spec.evaluate(session)
    results.atoms.residues.ribbon_displays = False

_unribbon_desc = cli.CmdDesc(optional=[("spec", atomspec.AtomSpecArg)],
                             synopsis='display ribbon for specified residues')


def set_cmd(session, bg_color=None, silhouettes=None):
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

from . import color
_set_desc = cli.CmdDesc(
    keyword=[('bg_color', color.ColorArg), ('silhouettes', cli.BoolArg)],
    synopsis="set preferences"
)

def style_command(session, atom_style, atoms = None):
    from .structure import AtomicStructure
    s = {'sphere':AtomicStructure.SPHERE_STYLE,
         'ball':AtomicStructure.BALL_STYLE,
         'stick':AtomicStructure.STICK_STYLE,
         }[atom_style.lower()]
    if atoms is None:
        for m in session.models.list():
            if isinstance(m, AtomicStructure):
                m.atoms.draw_modes = s
    else:
        asr = atoms.evaluate(session)
        asr.atoms.draw_modes = s

_style_desc = cli.CmdDesc(required = [('atom_style', cli.EnumOf(('sphere', 'ball', 'stick')))],
                          optional=[("atoms", atomspec.AtomSpecArg)],
                          synopsis='change atom depiction')

class CallForNFrames:
    # CallForNFrames acts like a function that keeps track of per-frame
    # functions.  But those functions need state, so that state is
    # encapsulated in instances of this class.
    #
    # Instances are automatically added to the given session 'Attribute'.

    Infinite = -1
    Attribute = 'motion_in_progress'    # session attribute

    def __init__(self, func, n, session):
        self.func = func
        self.n = n
        self.session = session
        self.frame = 0
        v = session.main_view
        v.add_callback('new frame', self)
        if not hasattr(session, self.Attribute):
            setattr(session, self.Attribute, set([self]))
        else:
            getattr(session, self.Attribute).add(self)

    def __call__(self):
        f = self.frame
        if self.n != self.Infinite and f >= self.n:
            self.done()
        else:
            self.func(self.session, f)
            self.frame = f + 1

    def done(self):
        s = self.session
        v = s.main_view
        v.remove_callback('new frame', self)
        getattr(s, self.Attribute).remove(self)


#
# Turn command to rotate models.
#
def turn(session, axis, angle, frames=None):
    v = session.main_view
    c = v.camera
    cv = c.position
    saxis = cv.apply_without_translation(axis)  # Convert axis from camera to scene coordinates
    center = v.center_of_rotation
    from .geometry import rotation
    r = rotation(saxis, -angle, center)
    if frames is None:
        c.position = r * cv
    else:
        def rotate(session, frame, r=r, c=c):
            c.position = r * c.position
        CallForNFrames(rotate, frames, session)

_turn_desc = cli.CmdDesc(
    required=[('axis', cli.AxisArg),
              ('angle', cli.FloatArg)],
    optional=[('frames', cli.PositiveIntArg)],
    synopsis='rotate models'
)


def roll(session, axis=(0, 1, 0), angle=1.5, frames=CallForNFrames.Infinite):
    turn(session, axis, angle, frames)

_roll_desc = cli.CmdDesc(
    optional=[('axis', cli.AxisArg),
              ('angle', cli.FloatArg),
              ('frames', cli.PositiveIntArg)],
    synopsis='rotate models'
)


def move(session, axis, distance, frames=None):
    v = session.main_view
    c = v.camera
    cv = c.position
    saxis = cv.apply_without_translation(axis)  # Convert axis from camera to scene coordinates
    from .geometry import translation
    t = translation(saxis * -distance)
    if frames is None:
        c.position = t * cv
    else:
        def translate(session, frame, t=t):
            c.position = t * c.position
        CallForNFrames(translate, frames, session)

_move_desc = cli.CmdDesc(
    required=[('axis', cli.AxisArg),
              ('distance', cli.FloatArg)],
    optional=[('frames', cli.PositiveIntArg)],
    synopsis='translate models'
)


def freeze(session):
    if not hasattr(session, CallForNFrames.Attribute):
        return
    for mip in tuple(getattr(session, CallForNFrames.Attribute)):
        mip.done()
_freeze_desc = cli.CmdDesc(synopsis='stop all motion')


def motion_in_progress(session):
    # Return True if there are non-infinite motions
    if not hasattr(session, CallForNFrames.Attribute):
        return False
    has_finite_motion = False
    for m in getattr(session, CallForNFrames.Attribute):
        if m.n == CallForNFrames.Infinite:
            return False
        has_finite_motion = True
    return has_finite_motion


def wait(session, frames=None):
    v = session.main_view
    if frames is None:
        # from ..commands.motion import motion_in_progress
        while motion_in_progress(session):
            v.redraw_needed = True  # Trigger frame rendered callbacks to cause image capture.
            v.draw(only_if_changed=True)
    else:
        for f in range(frames):
            v.redraw_needed = True  # Trigger frame rendered callbacks to cause image capture.
            v.draw(only_if_changed=True)
_wait_desc = cli.CmdDesc(
    optional=[('frames', cli.PositiveIntArg)],
    synopsis='suspend command processing for a specified number of frames'
             ' or until finite motions have stopped ')


def crossfade(session, frames=30):
    from .graphics import CrossFade
    CrossFade(session.main_view, frames)
_crossfade_desc = cli.CmdDesc(
    optional=[('frames', cli.PositiveIntArg)],
    synopsis='Fade between one rendered scene and the next scene.')


def register(session):
    """Register common cli commands"""
    cli.register('exit', _exit_desc, exit)
    cli.alias(session, "quit", "exit $*")
    cli.register('open', _open_desc, open)
    cli.register('close', _close_desc, close)
    cli.register('delete', _delete_desc, delete)
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
    cli.register('ribbon', _ribbon_desc, ribbon)
    cli.register('~ribbon', _unribbon_desc, unribbon)
    cli.register('set', _set_desc, set_cmd)
    cli.register('style', _style_desc, style_command)
    cli.register('turn', _turn_desc, turn)
    cli.register('roll', _roll_desc, roll)
    cli.register('move', _move_desc, move)
    cli.register('freeze', _freeze_desc, freeze)
    cli.register('wait', _wait_desc, wait)
    cli.register('crossfade', _crossfade_desc, crossfade)
    from . import molsurf
    molsurf.register_surface_command()
    molsurf.register_sasa_command()
    molsurf.register_buriedarea_command()
    from . import scolor
    scolor.register_scolor_command()
    from . import lightcmd
    lightcmd.register_lighting_command()
    lightcmd.register_material_command()
    from . import map
    map.register_volume_command()
    map.register_molmap_command()
    from .map import filter
    filter.register_vop_command()
    from .map import fit
    fit.register_fitmap_command()
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
    from . import crosslinks
    crosslinks.register_crosslink_command()
    from . import split
    split.register_split_command()
    from . import perframe
    perframe.register_perframe_command()
    from . import sym
    sym.register_sym_command()

    # def lighting_cmds():
    #     import .lighting.cmd as cmd
    #     cmd.register()
    # cli.delay_registration('lighting', lighting_cmds)

    from . import atomspec
    atomspec.register_selector(None, "sel", _sel_selector)
    atomspec.register_selector(None, "strands", _strands_selector)


def _sel_selector(session, models, results):
    from .structure import AtomicStructure
    for m in models:
        if m.any_part_selected():
            results.add_model(m)
            if isinstance(m, AtomicStructure):
                for atoms in m.selected_items('atoms'):
                    results.add_atoms(atoms)


def _strands_selector(session, models, results):
    from .structure import AtomicStructure
    for m in models:
        if isinstance(m, AtomicStructure):
            strands = m.residues.filter(m.residues.is_sheet)
            if strands:
                results.add_model(m)
                results.add_atoms(strands.atoms)
