# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2026 Regents of the University of California. All rights reserved.
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

"""
Faster entry point lookups while setuptools runs.

setuptools looks up its commands and setup keywords through
``importlib.metadata.entry_points``, a dozen or more times per build, and
each lookup reads the entry points of every installed distribution.  In
ChimeraX, with hundreds of distributions installed, that was most of the
time it takes to build a pure Python bundle.  Nothing is installed while a
wheel is built, so read them once.
"""

import importlib.metadata
from contextlib import contextmanager


@contextmanager
def cached_entry_points():
    original = importlib.metadata.entry_points
    all_entry_points = None

    def entry_points(**params):
        nonlocal all_entry_points
        if all_entry_points is None:
            all_entry_points = original()
        return all_entry_points.select(**params) if params else all_entry_points

    importlib.metadata.entry_points = entry_points
    try:
        yield
    finally:
        importlib.metadata.entry_points = original
