# vim: set expandtab ts=4 sw=4:

from chimera.core import cli

def _get_gui(session, create=False):
    from .gui import Log
    running = session.tools.find_by_class(Log)
    if len(running) > 1:
        raise RuntimeError("too many log instances running")
    if not running:
        if create:
            return Log(session)
        else:
            return None
    else:
        return running[0]

def hide(session):
    log = _get_gui(session)
    if log is not None:
        log.display(False)
hide_desc = cli.CmdDesc()

def show(session):
    log = _get_gui(session, create=True)
    if log is not None:
        log.display(True)
show_desc = cli.CmdDesc()

def test(session):
    session.logger.info("Something in <i>italics</i>!", is_html=True)
    session.logger.error("HTML <i>error</i> message", is_html=True)
    session.logger.warning("Plain text warning")
    from PIL import Image
    session.logger.info("axes",
        image=Image.open("/Users/pett/Documents/axes.png"))
    session.logger.info("Text after the image\nSecond line")
    session.logger.info("<pre>open xyzzy\n..........^\nMissing or unknown file type</pre>", is_html=True)
test_desc = cli.CmdDesc()
