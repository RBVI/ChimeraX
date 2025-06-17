# vim: set expandtab ts=4 sw=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This particular file is part of the ChimeraX library. You can also
# redistribute and/or modify it under the terms of the GNU Lesser General
# Public License version 2.1 as published by the Free Software Foundation.
# For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
# EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
# LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
# VERSION 2.1
#
# This notice must be embedded in or attached to all copies, including partial
# copies, of the software or any revisions or derivations thereof.
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

    # Allow Qt fatal errors to also be written to the faulthandler log file.
    if hasattr(session, 'ui') and hasattr(session.ui, 'set_fatal_error_log_file'):
        session.ui.set_fatal_error_log_file(_fault_handler_file)
        
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
def _show_bug_report_dialog(session, traceback, advise_on_common_crashes = True):
    from chimerax.bug_reporter import show_bug_reporter
    from chimerax.core.colors import scheme_color
    color = scheme_color('error')
    known_crash = False
    advice = '<p>Please describe steps that led to the crash here.</p>'
    if advise_on_common_crashes:
        ccm = _common_crash_message(traceback)
        if ccm is not None:
            advice = f'<p><font color="{color}">{ccm}</font></p>'
            known_crash = True
    br = show_bug_reporter(session, is_known_crash = known_crash)
    msg = (f'<h3><font color="{color}">Last time you used ChimeraX it crashed.</font></h3>'
           f'{advice}'
           f'<pre>\n{traceback}\n</pre>')
    br.set_description(msg, minimum_height = 200)

# -----------------------------------------------------------------------------
#
def _common_crash_message(traceback):
    if 'Graphics hardware encountered an error and was reset' in traceback:
        msg = 'This is an Apple Intel or AMD graphics driver crash that may be related to showing a scene that is complex and uses too much graphics memory.  Apple is unlikely to ever fix this since they no longer make computers with Intel or AMD graphics.  The crash does not happen with Apple M1,M2,M3... graphics.'

    elif 'Qt fatal error: Failed to initialize graphics backend for OpenGL' in traceback:
        msg = 'The Qt window toolkit was unable to start because it could not initialize OpenGL graphics.  This can happen on Linux when using remote display or when a defective graphics driver is installed.  Remote display of ChimeraX is not supported due to the many issues with remote display of OpenGL graphics.  If you are not using remote display you need to update your computer graphics driver.'

    elif '_NSViewHierarchyDidChangeBackingProperties' in traceback or "displayConfigFinalizedProc" in traceback:
        msg = 'The Qt window toolkit crashed due to a display configuration change, typically when waking from sleep or when an external display is disconnected or connected.  This has only been seen on Mac computers.  We hope a newer version of Qt will fix it.  We update ChimeraX daily builds whenever a new Qt is released.  You can check here <a href="https://www.cgl.ucsf.edu/chimerax/docs/troubleshoot.html#macdisplay">https://www.cgl.ucsf.edu/chimerax/docs/troubleshoot.html#macdisplay</a> to see if it has been fixed in a newer ChimeraX.'

    else:
        msg = None

    if msg is None:
        return None

    advice = f'This is a known crash that we are unable to fix.  Here is information that may help you avoid this crash. {msg}'

    return advice
