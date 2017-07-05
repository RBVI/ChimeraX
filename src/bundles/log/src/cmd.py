# vim: set expandtab ts=4 sw=4:

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

def get_singleton(session, create=False):
    if not session.ui.is_gui:
        return None
    from chimerax.core import tools
    from .tool import Log
    return tools.get_singleton(session, Log, 'Log', create=create)

def log(session, save_path = None,
        thumbnail = False, text = None, html = None, width = 100, height = 100,
        warning_dialog = None, error_dialog = None):
    '''Operations on the Log window.

    Parameters
    ----------
    save_path : string
      Save log contents as html to a file.
    thumbnail : bool
      Place a thumbnail image of the current graphics in the log.
    height : int
      Height in pixels of thumbnail image.
    width : int
      Width in pixels of thumbnail image.
    warning_dialog : bool
      If true, warnings popup a separate dialog, if false no warning dialog is shown.
      In either case the warning appears in the log text.
    error_dialog : bool
      If true, errors popup a separate dialog, if false no error dialog is shown.
      In either case the errors appears in the log text.
    '''
    log = get_singleton(session, create = False)
    if log is not None:
        if not save_path is None:
            log.save(save_path)
        if thumbnail:
            im = session.main_view.image(width, height)
            log.log(log.LEVEL_INFO, 'graphics image', (im, True), True)
        if text:
            log.log(log.LEVEL_INFO, text, (None, False), False)
        if html:
            log.log(log.LEVEL_INFO, html, (None, False), True)
        if not warning_dialog is None:
            log.warning_shows_dialog = warning_dialog
        if not error_dialog is None:
            log.error_shows_dialog = error_dialog
    else:
        log = session.logger
        if not save_path is None:
            log.warning("no log tool to save")
        if thumbnail:
            log.warning("no log tool for thumbnail")
        if text:
            log.info(text)
        if html:
            log.info(html, is_html=True)
        if not warning_dialog is None:
            pass
        if not error_dialog is None:
            pass

def log_show(session):
    '''Show the log window.'''
    log = get_singleton(session, create = True)
    log.display(True)

def log_hide(session):
    '''Hide the log window.'''
    log = get_singleton(session, create = False)
    if log:
        log.display(False)

def log_clear(session):
    '''Clear the log window.'''
    log = get_singleton(session, create = False)
    if log:
        log.clear()

def register_log_command(logger):
    from chimerax.core.commands import register, CmdDesc, NoArg, BoolArg, IntArg, RestOfLine, SaveFileNameArg
    log_desc = CmdDesc(keyword = [('thumbnail', NoArg),
                                  ('text', RestOfLine),
                                  ('html', RestOfLine),
                                  ('width', IntArg),
                                  ('height', IntArg),
                                  ('save_path', SaveFileNameArg),
                                  ('warning_dialog', BoolArg),
                                  ('error_dialog', BoolArg)],
                       synopsis = 'Added text or thumbnail images to the log panel'
    )
    register('log', log_desc, log, logger=logger)
    register('log show', CmdDesc(synopsis='Show log panel'), log_show, logger=logger)
    register('log hide', CmdDesc(synopsis='Hide log panel'), log_hide, logger=logger)
    register('log clear', CmdDesc(synopsis='Clear log panel'), log_clear, logger=logger)
