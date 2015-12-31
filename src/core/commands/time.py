# vim: set expandtab shiftwidth=4 softtabstop=4:


def time(session, command):
    '''Time a command.'''
    from time import time
    t0 = time()
    from .run import run
    run(session, command)
    t1 = time()

    session.triggers.add_handler('frame drawn', lambda name,data,s=session,t=t1: _report_draw_time(s,t))

    msg = 'command time %.4g seconds' % (t1-t0)
    log = session.logger
    log.status(msg)
    log.info(msg)

def _report_draw_time(session, tstart):
    from time import time
    tend = time()
    session.logger.info('draw time %.4g seconds' % (tend-tstart))
    from ..triggerset import DEREGISTER
    return DEREGISTER

def register_command(session):
    from . import CmdDesc, register, RestOfLine
    desc = CmdDesc(required=[('command', RestOfLine)],
                   synopsis='time a command')
    register('time', desc, time)
