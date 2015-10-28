# vim: set expandtab ts=4 sw=4:
"""
errors: define Chimera-specific errors
======================================

"""

class CancelOperation(Exception):
    """User-requested cancellation of a long-running task"""
    pass

class NotABug(Exception):
    """Base class for errors that shouldn't produce tracebacks/bug reports"""
    pass

class LimitationError(NotABug):
    """Known implementation limitation"""
    pass

class NonChimeraError(NotABug):
    """Error whose cause is outside Chimera, for example a temporary network error"""
    pass

class UserError(NotABug):
    """User provided wrong input, or took a wrong action"""
    pass

class ChimeraSystemExit(SystemExit):
    """Like SystemExit, but tools can clean up before exit"""
    pass
