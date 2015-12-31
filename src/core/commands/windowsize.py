# vim: set expandtab shiftwidth=4 softtabstop=4:


def windowsize(session, width=None, height=None):
    '''Report or set graphics window size in pixels.'''

    v = session.main_view
    w,h = v.window_size
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
        win = session.ui.main_window
        cs = win.GetSize()
        cww,cwh = cs.GetWidth(), cs.GetHeight()
        ww = cww + (width-w)
        wh = cwh + (height-h)
        win.SetSize(ww,wh)

def register_command(session):
    from . import CmdDesc, register, IntArg
    desc = CmdDesc(optional=[('width', IntArg),
                             ('height', IntArg)],
                   synopsis='report or set window size')
    register('windowsize', desc, windowsize)
