# vim: set expandtab shiftwidth=4 softtabstop=4:

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
import os

def path_to_src() -> str:
    return os.path.dirname(__file__)

def get_lib() -> str:
    return path_to_src()

def get_include() -> str:
    return os.path.join(path_to_src(), "include")

from . import _sw as SmithWaterman

NEEDLEMAN_WUNSCH = "Needleman-Wunsch"
SMITH_WATERMAN = "Smith-Waterman"
