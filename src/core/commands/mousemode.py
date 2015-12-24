# vim: set expandtab shiftwidth=4 softtabstop=4:

def mousemode(session, left_mode=None, middle_mode=None, right_mode=None,
              wheel_mode=None, pause_mode=None,
              alt=None, command=None, control=None, shift=None):
    '''
    Set or list mouse modes.  If no options given lists mouse modes.

    Parameters
    ----------
    left_mode : mode
       Bind the left mouse button to mouse mode.
    middle_mode : mode
       Bind the middle mouse button to mouse mode.
    right_mode : mode
       Bind the right mouse button to mouse mode.
    wheel_mode : mode
       Bind the mouse wheel to mouse mode.
    pause_mode : mode
       Bind mouse hover to mouse mode.
    '''
    mm = session.ui.main_window.graphics_window.mouse_modes
    bmode = (('left', left_mode), ('middle', middle_mode), ('right', right_mode),
             ('wheel', wheel_mode), ('pause', pause_mode))
    modifiers = [n for s,n in [(alt,'alt'), (command,'command'), (control,'control'), (shift,'shift')] if s]
    for button, mode in bmode:
        if mode is not None:
            mm.bind_mouse_mode(button, modifiers, mode)
            
    # List current modes.
    if len([b for b,m in bmode if m is not None]) == 0:
        lines = []
        order = {'left':1, 'middle':2, 'right':3, 'wheel':4}
        for b in sorted(list(mm.bindings), key = lambda b: order.get(b.button,5)):
            mline = ' %s: %s' % (b.button,b.mode.name)
            if b.modifiers:
                mline = '%s %s' % (','.join(b.modifiers), mline)
            lines.append(mline)
        lines.append('Available modes: %s' % ', '.join(m.name for m in mm.modes))
        msg = '\n'.join(lines)
        session.logger.info(msg)

from .cli import Annotation
class MouseModeArg(Annotation):
    '''Annotation for specifying a mouse mode.'''

    def __init__(self, session):
        Annotation.__init__(self)
        self._session = session

    def parse(self, text, session):
        from ..ui import mousemodes
        modes = session.ui.main_window.graphics_window.mouse_modes.modes
        from .cli import EnumOf
        mode_arg = EnumOf(tuple(m.name for m in modes))
        value, used, rest = mode_arg.parse(text, session)
        mmap = {m.name:m for m in modes}
        return mmap[value], used, rest

    @property
    def name(self):
        modes = self._session.ui.main_window.graphics_window.mouse_modes.modes
        return 'one of ' + ', '.join("'%s'" % m.name for m in modes)

def register_command(session):
    from .cli import CmdDesc, register, StringArg, NoArg
    mode_arg = MouseModeArg(session)
    desc = CmdDesc(
        keyword=[
            ('left_mode', mode_arg),
            ('middle_mode', mode_arg),
            ('right_mode', mode_arg),
            ('wheel_mode', mode_arg),
            ('pause_mode', mode_arg),
            ('alt', NoArg),
            ('command', NoArg),
            ('control', NoArg),
            ('shift', NoArg),
        ],
        synopsis='set mouse mode'
    )
    register('mousemode', desc, mousemode)
