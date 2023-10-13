# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

# -----------------------------------------------------------------------------
# Implementation of "ui mousemode" command.
#
def register_mousemode_command(logger):

    from chimerax.core.commands import CmdDesc, register, create_alias, NoArg, FloatArg

    mode_arg = MouseModeArg(logger.session)
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
        synopsis='set mouse mode',
        url="help:user/commands/ui.html#mousemode"
    )
    register('mousemode', desc, mousemode, logger=logger)
    create_alias('ui mousemode', 'mousemode $*', logger=logger,
                 url="help:user/commands/ui.html#mousemode")

    desc = CmdDesc(
        optional = [('mode', mode_arg)],
        keyword=[('speed', FloatArg)],
        synopsis='set a mouse mode parameter',
        url="help:user/commands/ui.html#mousemode"
    )
    register('mousemode setting', desc, mousemode_setting, logger=logger)
    create_alias('ui mousemode setting', 'mousemode setting $*', logger=logger,
                 url="help:user/commands/ui.html#mousemode")

# -----------------------------------------------------------------------------
#
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
    if not session.ui.is_gui:
        session.logger.info("mouse is not supported in nogui mode")
        return
    mm = session.ui.mouse_modes
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

# -----------------------------------------------------------------------------
#
def mousemode_setting(session, mode = None, speed = None):
    '''
    Set a mouse mode parameter.

    Parameters
    ----------
    mode : mode
       The mouse mode to set parameter.  If no mode is given then the parameter is
       set for all modes for which that parameter can be set.
    speed : float
       Sensitivity to mouse motion.  Default 1.  Not all modes support this setting.
    '''
    if not session.ui.is_gui:
        session.logger.info("mouse is not supported in nogui mode")
        return

    if mode is None:
        modes = [m for m in session.ui.mouse_modes.modes if hasattr(m, 'speed')]
    elif hasattr(mode, 'speed'):
        modes = [mode]
    else:
        msg = 'Mouse mode %s does not support speed adjustment' % mode.name
        session.logger.warning(msg)
        modes = []
        
    if speed is None:
        speeds = ', '.join('%s speed = %.3g' % (mode.name, mode.speed) for mode in modes)
        msg = 'Mouse mode %s' % speeds
        session.logger.info(msg)
    else:
        for mode in modes:
            mode.speed = speed

# -----------------------------------------------------------------------------
#
from chimerax.core.commands import Annotation
class MouseModeArg(Annotation):
    '''Annotation for specifying a mouse mode.'''

    def __init__(self, session):
        Annotation.__init__(self)
        self._session = session

    @property
    def name(self):
        if not self._session.ui.is_gui:
            return 'a mouse mode'
        from html import escape
        modes = self._session.ui.mouse_modes.modes
        return 'one of ' + ', '.join("'%s'" % escape(m.name) for m in modes)

    @property
    def _html_name(self):
        if not self._session.ui.is_gui:
            return 'a mouse mode'
        from html import escape
        modes = self._session.ui.mouse_modes.modes
        return 'one of ' + ', '.join("<b>%s</b>" % escape(m.name) for m in modes)

    def parse(self, text, session):
        modes = session.ui.mouse_modes.modes
        from chimerax.core.commands import EnumOf
        mode_arg = EnumOf(tuple(m.name for m in modes))
        value, used, rest = mode_arg.parse(text, session)
        mmap = {m.name:m for m in modes}
        return mmap[value], used, rest
