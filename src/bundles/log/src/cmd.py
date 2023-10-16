# vim: set expandtab ts=4 sw=4:

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

def get_singleton(session, create=False):
    if not session.ui.is_gui:
        return None
    from chimerax.core import tools
    from .tool import Log
    return tools.get_singleton(session, Log, 'Log', create=create)

def log(session, thumbnail = False, text = None, html = None, width = 100, height = 100,
        warning_dialog = None, error_dialog = None):
    '''Operations on the Log window.

    Parameters
    ----------
    text : string
       Text to show in the log.
    html : bool
       Indicate if the text is HTML text.
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
        if thumbnail:
            im = session.main_view.image(width, height)
            log.log(log.LEVEL_INFO, 'graphics image', (im, True), True)
        if text:
            log.log(log.LEVEL_INFO, text + '\n', (None, False), False)
        if html:
            log.log(log.LEVEL_INFO, html, (None, False), True)
        if warning_dialog is not None:
            log.warning_shows_dialog = warning_dialog
        if error_dialog is not None:
            log.error_shows_dialog = error_dialog
    else:
        log = session.logger
        if thumbnail:
            log.warning("no log tool for thumbnail")
        if text:
            log.info(text)
        if html:
            log.info(html, is_html=True)
        if warning_dialog is not None:
            pass
        if error_dialog is not None:
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

def log_save(session, file_name, executable_links = None):
    '''Save the log window

    Parameters
    ----------
    file_name : string
      Save log contents as html to a file.
    executable_links: bool or None
      Whether links should execute command or show command help.  If None, use current log setting.
    '''
    log = get_singleton(session, create = False)
    if log is not None:
        log.save(file_name, executable_links=executable_links)
    else:
        log = session.logger
        log.warning("no log tool to save")

def log_settings(session, error_dialog = None, warning_dialog = None):
    '''Save the log window

    Parameters
    ----------
    error_dialog : bool
      Whether to show errors in a separate dialog (for the remainder of this session)
    warning_dialog : bool
      Whether to show warnings in a separate dialog (for the remainder of this session)
    '''
    from .settings import settings
    if error_dialog is not None:
        settings.errors_raise_dialog = error_dialog

    if warning_dialog is not None:
        settings.warnings_raise_dialog = warning_dialog

def log_metadata(session, models=None, verbose=False):
    if models is None:
        models = session.models
    log = get_singleton(session, create = False)
    if log is not None:
        any_metadata = False
        for model in models:
            if model.has_formatted_metadata(session):
                any_metadata = True
                model.show_metadata(session, verbose=verbose, log=log)
        if not any_metadata:
            if not models:
                log.log(log.LEVEL_INFO, "No models match specifier", (None, False), False)
            elif len(models) == 1:
                log.log(log.LEVEL_INFO, "The model has no metadata", (None, False), False)
            else:
                log.log(log.LEVEL_INFO, "No models had metadata", (None, False), False)
    else:
        session.logger.warning("no log tool for metadata")

def register_log_command(logger):
    from chimerax.core.commands import register, CmdDesc, NoArg, BoolArg, IntArg, RestOfLine, \
        SaveFileNameArg, ModelsArg
    log_desc = CmdDesc(keyword = [('thumbnail', NoArg),
                                  ('text', RestOfLine),
                                  ('html', RestOfLine),
                                  ('width', IntArg),
                                  ('height', IntArg)],
                       synopsis = 'Add text or thumbnail images to the log'
    )
    register('log', log_desc, log, logger=logger)
    register('log show', CmdDesc(synopsis='Show log panel'), log_show, logger=logger)
    register('log hide', CmdDesc(synopsis='Hide log panel'), log_hide, logger=logger)
    register('log clear', CmdDesc(synopsis='Clear log panel'), log_clear, logger=logger)
    save_desc = CmdDesc(
                        required = [('file_name', SaveFileNameArg)],
                        keyword = [('executable_links', BoolArg)],
                        synopsis = 'Save log to file'
    )
    register('log save', save_desc, log_save, logger=logger)
    settings_desc = CmdDesc(
                        keyword = [('warning_dialog', BoolArg), ('error_dialog', BoolArg)],
                        synopsis = 'Temporarily change log settings'
    )
    register('log settings', settings_desc, log_settings, logger=logger)
    metadata_desc = CmdDesc(
                        optional = [('models', ModelsArg)],
                        keyword = [('verbose', BoolArg)],
                        synopsis = 'Add structure metadata table to the log'
    )
    register('log metadata', metadata_desc, log_metadata, logger=logger)
