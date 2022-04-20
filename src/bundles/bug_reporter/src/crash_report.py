# vim: set expandtab ts=4 sw=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

# -----------------------------------------------------------------------------
#
def check_for_crash(session):
    import sys
    if sys.platform == 'darwin':
        from .mac_crash_report import check_for_crash_on_mac
        traceback = check_for_crash_on_mac(session)
        if traceback:
            # Add Python traceback if available.
            python_traceback = _python_fault_handler_traceback()
            if python_traceback:
                traceback = '%s\n\n%s' % (python_traceback, traceback)
    else:
        # On Windows and Linux only have python traceback.
        traceback = _python_fault_handler_traceback()

    if traceback:
        traceback += _last_log_text()
        # Delay showing bug report dialog until ChimeraX fully started.
        def _report(trigger_name, update_loop, session=session, traceback = traceback):
            _show_bug_report_dialog(session, traceback)
            from chimerax.core import triggerset
            return triggerset.DEREGISTER
        session.triggers.add_handler('new frame', _report)

# -----------------------------------------------------------------------------
#
def register_signal_handler(session):
    '''
    Use Python faulthandler module to write Python traceback if we crash.
    Then next time ChimeraX starts if that file exists show report bug tool.
    '''
    try:
        traceback_path = _python_traceback_file_path()
        traceback_file = open(traceback_path, 'w')
    except IOError:
        # If we can't write the file just do without.
        return

    # Remove the file at exit if no crash.
    def remove_crash_file(file=traceback_file, path=traceback_path):
        import os
        try:
            file.close()
            os.remove(path)
        except Exception:
            pass
    # Python atexit routines are not called on a fatal signal.
    import atexit
    atexit.register(remove_crash_file)
    
    import faulthandler
    faulthandler.enable(traceback_file)

    global _fault_handler_file
    _fault_handler_file = traceback_file
    
# -----------------------------------------------------------------------------
#
_fault_handler_file = None
def clear_fault_handler_file(session):
    if _fault_handler_file is not None:
        try:
            _fault_handler_file.truncate(0)
        except Exception:
            pass
    
# -----------------------------------------------------------------------------
#
def _python_fault_handler_traceback(traceback_path = None, remove_file = True):
    '''
    Return Python traceback from previous crash as a string or None.
    '''
    if traceback_path is None:
        traceback_path = _python_traceback_file_path()
        
    from os.path import getsize, isfile
    if not isfile(traceback_path) or getsize(traceback_path) == 0:
        return None

    try:
        f = open(traceback_path, 'r')
        traceback = f.read()
        f.close()
        if remove_file:
            from os import remove
            remove(traceback_path)
    except IOError:
        return None

    return traceback

# -----------------------------------------------------------------------------
#
def _python_traceback_file_path():
    from chimerax import app_dirs_unversioned
    import os.path
    traceback_path = os.path.join(app_dirs_unversioned.user_config_dir, 'crash_traceback.txt')
    return traceback_path

# -----------------------------------------------------------------------------
#
def register_log_recorder(session):
    '''
    Save log messages to a file so they can be reported in case of crash.
    '''

    # Need to delay since log panel has not yet been started.
    def _delayed_register_log_recorder(trigger_name, update_loop, session=session):
        _register_log_recorder(session)
        from chimerax.core import triggerset
        return triggerset.DEREGISTER

    session.triggers.add_handler('new frame', _delayed_register_log_recorder)

# -----------------------------------------------------------------------------
#
def _register_log_recorder(session):

    from chimerax.log.cmd import get_singleton
    log = get_singleton(session)
    if log is None:
        return

    try:
        log_path = _last_log_file_path()
        log_file = open(log_path, 'w')
    except IOError:
        # If we can't write the file just do without.
        return

    # Remove the file at exit if no crash.
    def remove_log_file(file=log_file, path=log_path):
        import os
        try:
            file.close()
            os.remove(path)
        except Exception:
            pass
    # Python atexit routines are not called on a fatal signal.
    import atexit
    atexit.register(remove_log_file)
    
    log.record_to_file(log_file)

# -----------------------------------------------------------------------------
#
def _last_log_text():
    '''
    Return log text from last session.
    '''
    last_log_path = _last_log_file_path()

    try:
        f = open(last_log_path, 'r')
        log = f.read()
        f.close()
    except IOError:
        return ''

    try:
        from chimerax.log.tool import log_html_to_plain_text
        text = log_html_to_plain_text(log)
    except BaseException:
        return ''

    text = '===== Log before crash start =====\n' + text + '\n===== Log before crash end ====='
    return text

# -----------------------------------------------------------------------------
#
def _last_log_file_path():
    from chimerax import app_dirs_unversioned
    import os.path
    last_log_path = os.path.join(app_dirs_unversioned.user_config_dir, 'last_log.txt')
    return last_log_path

# -----------------------------------------------------------------------------
#
def _show_bug_report_dialog(session, traceback):
    from chimerax.bug_reporter import show_bug_reporter
    br = show_bug_reporter(session)
    msg = ('<p><font color=red>Last time you used ChimeraX it crashed.</font><br>'
           'Please describe steps that led to the crash here.</p>'
           '<pre>\n%s\n</pre>' % traceback)
    br.set_description(msg)
