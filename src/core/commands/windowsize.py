# vim: set expandtab shiftwidth=4 softtabstop=4:


def window_size(session, width=None, height=None):
    '''Report or set graphics window size in pixels.'''

    v = session.main_view
    w, h = v.window_size
    if width is None and height is None:
        msg = 'window size %d %d' % (w, h)
        log = session.logger
        log.status(msg)
        log.info(msg)
    else:
        if width is None:
            width = w
        if height is None:
            height = h
        if not session.ui.is_gui:
            v.window_size = width, height
        else:
            session.ui.main_window.adjust_size(width-w, height-h)

def register_command(session):
    from . import CmdDesc, register, PositiveIntArg
    desc = CmdDesc(optional=[('width', PositiveIntArg),
                             ('height', PositiveIntArg)],
                   synopsis='report or set window size')
    register('windowsize', desc, window_size)
