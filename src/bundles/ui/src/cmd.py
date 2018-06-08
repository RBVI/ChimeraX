# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

# -----------------------------------------------------------------------------
# Implementation of "ui" command.
#
def register_ui_command(logger):

    from chimerax.core.commands import CmdDesc, register, create_alias
    from chimerax.core.commands import BoolArg, StringArg, NoArg

    ui_autostart_desc = CmdDesc(
        required=[('do_start', BoolArg), ('tool_name', StringArg)],
        synopsis = 'control whether a tool is launched at ChimeraX startup')
    register('ui autostart', ui_autostart_desc, ui_autostart, logger=logger)

    ui_dockable_desc = CmdDesc(
        required=[('dockable', BoolArg), ('tool_name', StringArg)],
        synopsis = "control whether a tool's windows can be docked")
    register('ui dockable', ui_dockable_desc, ui_dockable, logger=logger)

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
        synopsis='set mouse mode'
    )
    register('ui mousemode', desc, ui_mousemode, logger=logger)
    create_alias('mousemode', 'ui mousemode $*', logger=logger)

# -----------------------------------------------------------------------------
#
def ui_autostart(session, do_start, tool_name):
    '''
    Control whether a tool is launched at ChimeraX startup
    '''

    settings = session.ui.settings
    autostart = settings.autostart[:]
    if do_start:
        if tool_name not in autostart:
            autostart.append(tool_name)
    else:
        while tool_name in autostart:
            autostart.remove(tool_name)
    settings.autostart = autostart
    settings.save()

# -----------------------------------------------------------------------------
#
def ui_dockable(session, dockable, tool_name):
    '''
    Control whether a tool's windows are dockable
    '''

    settings = session.ui.settings
    undockable_tools = settings.undockable[:]
    if dockable:
        while tool_name in undockable_tools:
            undockable_tools.remove(tool_name)
    else:
        if tool_name not in undockable_tools:
            undockable_tools.append(tool_name)
    settings.undockable = undockable_tools
    settings.save()
    if session.ui.is_gui:
        session.ui.main_window._dockability_change(tool_name, dockable)

def ui_mousemode(session, left_mode=None, middle_mode=None, right_mode=None,
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
