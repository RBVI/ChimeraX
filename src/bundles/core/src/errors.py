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
errors: define ChimeraX-specific errors
=======================================

"""

class CancelOperation(Exception):
    """User-requested cancellation of a task"""
    pass

class NotABug(Exception):
    """Base class for errors that shouldn't produce tracebacks/bug reports"""
    pass

class LimitationError(NotABug):
    """Known implementation limitation"""
    pass

class NonChimeraError(NotABug):
    """Error whose cause is outside ChimeraX, for example a temporary network error"""
    pass
# allow NonChimeraXError as well...
NonChimeraXError = NonChimeraError

class UserError(NotABug):
    """User provided wrong input, or took a wrong action"""
    pass

class ChimeraSystemExit(SystemExit):
    """Like SystemExit, but tools can clean up before exit"""
    pass
