# vim: set expandtab shiftwidth=4 softtabstop=4:

def mousemode(session, left_mode=None, middle_mode=None, right_mode=None,
              wheel_mode=None, pause_mode=None):
    '''
    Set or list mouse modes.  If no options given lists mouse modes.

    Parameters
    ----------
    left_mode : mode name
       Bind the left mouse button to the named mouse mode.
    middle_mode : mode name
       Bind the middle mouse button to the named mouse mode.
    right_mode : mode name
       Bind the right mouse button to the named mouse mode.
    wheel_mode : mode name
       Bind the mouse wheel to the named mouse mode.
    pause_mode : mode name
       Bind mouse hover to the named mouse mode.
    '''
    mm = session.ui.main_window.graphics_window.mouse_modes
    modes = {m.name:m for m in mm.modes}
    bmode = (('left', left_mode), ('middle', middle_mode), ('right', right_mode),
             ('wheel', wheel_mode), ('pause', pause_mode))
    for button, mode_name in bmode:
        if mode_name is not None:
            m = _find_mode_by_name(mode_name, mm.modes)
            mm.bind_mouse_mode(button, m)
            
    # List current modes.
    if len([b for b,m in bmode if m is not None]) == 0:
        lines = ['Current mouse modes:']
        order = {'left':1, 'middle':2, 'right':3, 'wheel':4}
        for b, m in sorted(list(mm.bindings.items()), key = lambda bm: order.get(bm[0],5)):
            lines.append(' %s: %s' % (b,m.name))
        lines.append('Available modes: %s' % ', '.join(m.name for m in mm.modes))
        msg = '\n'.join(lines)
        session.logger.info(msg)

def _find_mode_by_name(name, modes):
    for m in modes:
        if m.name == name:
            return m
    matches = [m for m in modes if m.name.startswith(name)]
    if len(matches) != 1:
        from . import AnnotationError
        if len(matches) == 0:
            raise AnnotationError('Unknown mouse mode "%s", available modes: %s'
                                  % (mode_name, ', '.join(m.name for m in modes)))
        else:
            raise AnnotationError('Multiple modes match "%s": %s'
                                  % (mode_name, ', '.join(m.name for m in matches)))
    return matches[0]
            
def register_command(session):
    from .cli import CmdDesc, register, StringArg
    desc = CmdDesc(
        keyword=[
            ('left_mode', StringArg),
            ('middle_mode', StringArg),
            ('right_mode', StringArg),
            ('wheel_mode', StringArg),
            ('pause_mode', StringArg),
        ],
        synopsis='set mouse mode'
    )
    register('mousemode', desc, mousemode)
