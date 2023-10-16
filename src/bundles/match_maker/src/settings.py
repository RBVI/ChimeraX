# vim: set expandtab shiftwidth=4 softtabstop=4:

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

from .match import CP_BEST_BEST, AA_NEEDLEMAN_WUNSCH

defaults = {
    'chain_pairing': CP_BEST_BEST,
    'alignment_algorithm': AA_NEEDLEMAN_WUNSCH,
    'show_alignment': False,
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
    'overwrite_ss': False,
    'verbose_logging': False,
    'log_transformation_matrix': False,
}

from  chimerax.core.settings import Settings
from copy import deepcopy

class _MatchmakerSettings(Settings):
    EXPLICIT_SAVE = deepcopy(defaults)

# for the GUI
_settings = None
def get_settings(session):
    global _settings
    # don't initialize a zillion times, which would also overwrite
    # any changed but not saved settings
    if _settings is None:
        _settings = _MatchmakerSettings(session, "matchmaker")
    return _settings
