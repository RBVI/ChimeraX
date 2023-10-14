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
