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

# -----------------------------------------------------------------------------
# Check Mac crash log to see if ChimeraX crashed since the last time it was
# started.  If so display the ChimeraX bug report dialog with crash log info
# filled in.
#
def check_for_crash_on_mac(session):

    import sys
    if sys.platform != 'darwin':
        return None # Only check for crashes on Mac OS.

    # Get time of last check for crash logs.
    from .settings import BugReporterSettings
    settings = BugReporterSettings(session, 'Bug Reporter')
    last = settings.last_crash_check
    from time import time
    settings.last_crash_check = time()
    if last is None:
        return None         # No previous crash check time available.

    report = recent_chimera_crash(last)

    return report

# -----------------------------------------------------------------------------
#
def recent_chimera_crash(time):

    # Check if Mac Python crash log exists and was modified since last
    # time Chimera was started.
    dir = crash_logs_directory()
    if dir is None:
        return None

    log = recent_crash(time, dir, 'ChimeraX')
    return log

# -----------------------------------------------------------------------------
# On Mac OS 10.6 and later uses ~/Library/Logs/DiagnosticReports for crash logs.
#
def crash_logs_directory():

    from os.path import expanduser, isdir
    logd = expanduser('~/Library/Logs/DiagnosticReports')
    if isdir(logd):
        return logd
    return None

# -----------------------------------------------------------------------------
#
def recent_crash(time, dir, file_prefix):

    from os import listdir
    try:
        filenames = listdir(dir)
    except PermissionError:
        # Crash directory is not readable so can't report crashes.
        return None

    from os.path import getmtime, join
    pypaths = [join(dir,f) for f in filenames if f.startswith(file_prefix)]
    tpaths = [(getmtime(p), p) for p in pypaths]
    if len(tpaths) == 0:
        return None

    tpaths.sort()
    t, p = tpaths[-1]
    if t < time:
        return None     # No file more recent than time.

    f = open(p, 'r', encoding = 'iso-8859-1')
    log = f.read()
    f.close()

    return log
