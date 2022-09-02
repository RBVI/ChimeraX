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

"""
gui: Main ChimeraX graphical user interface
===========================================

The principal class that tool writers will use from this module is
:py:class:`MainToolWindow`, which is either instantiated directly, or
subclassed and instantiated to create the tool's main window.
Additional windows are created by calling that instance's
:py:meth:`MainToolWindow.create_child_window` method.

Rarely, methods are used from the :py:class:`UI` class to get
keystrokes typed to the main graphics window, or to execute code
in a thread-safe manner.  The UI instance is accessed as session.ui.
"""
import sys

class LogStdout:

        # Qt's error logging looks at the encoding of sys.stderr...
        encoding = 'utf-8'

        def __init__(self, logger):
            self.logger = logger
            self.closed = False
            self.errors = "ignore"

        def write(self, s):
            message = s
            message_level = 'INFO'
            # Assume that all logs will use the message format from
            # core.__init__ which is level:message
            need_message_level = len(s.split(':')) > 1
            if need_message_level:
                message_level = s.split(':')[0]
                # Messages may have colons in them, so maybe merge everything
                # from field 1 on.
                message = ':'.join(s.split(':')[1:])
            # TODO: Come up with something good for BUG
            if message_level == 'WARNING':
                self.logger.session.ui.thread_safe(
                    self.logger.warning, message, add_newline = False
                )
            elif message_level == 'ERROR':
                self.logger.session.ui.thread_safe(
                    self.logger.error, message, add_newline = False
                )
            else:
                self.logger.session.ui.thread_safe(
                    self.logger.info, message, add_newline = False
                )

        def flush(self):
            return

        def isatty(self):
            return False

LogStderr = LogStdout
sys.orig_stdout = sys.stdout
sys.orig_stderr = sys.stderr
sys.stdout = None
sys.stderr = None
