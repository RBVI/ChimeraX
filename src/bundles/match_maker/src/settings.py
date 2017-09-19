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

from . import CP_BEST_BEST, AA_NEEDLEMAN_WUNSCH

defaults = {
    'chain_pairing': CP_BEST_BEST,
    'alignment_algorithm': AA_NEEDLEMAN_WUNSCH,
    'show_sequence': False,
    'matrix': "BLOSUM-62",
    'gap_open': 12,
    'gap_extend': 1,
    'use_ss': True,
    'ss_mixture': 0.3,
    'ss_scores': {
        ('H', 'H'): 6.0,
        ('S', 'S'): 6.0,
        ('O', 'O'): 4.0,
        ('S', 'H'): -9.0,
        ('H', 'S'): -9.0,
        ('S', 'O'): -6.0,
        ('O', 'S'): -6.0,
        ('H', 'O'): -6.0,
        ('O', 'H'): -6.0
    },
    'iterate': True,
    'iter_cutoff': 2.0,
    'helix_open': 18,
    'strand_open': 18,
    'other_open': 6,
    'compute_ss': True,
}

from  chimerax.core.settings import Settings
from copy import deepcopy

class _MatchmakerSettings(Settings):
    EXPLICIT_SAVE = deepcopy(defaults)

# once a GUI is implemented it will need to call this...
settings = None
def init(session):
    global settings
    # don't initialize a zillion times, which would also overwrite
    # any changed but not saved settings
    if settings is None:
        settings = _MatchmakerSettings(session, "matchmaker")
