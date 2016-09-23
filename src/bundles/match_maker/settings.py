# vim: set expandtab shiftwidth=4 softtabstop=4:

from . import CP_BEST, AA_NEEDLEMAN_WUNSCH

defaults = {
	'chain_pairing': CP_BEST,
	'alignment_algorithm': AA_NEEDLEMAN_WUNSCH,
	'show_sequence': False,
	'matrix': "BLOSUM-62",
	'gap_open': 12,
	'gap_extend': 1,
	'use_ss': True,
	'ss_mixture': 0.3,
	'ss_scores': {
		('H', 'H'): 6,
		('S', 'S'): 6,
		('O', 'O'): 4,
		('S', 'H'): -9,
		('H', 'S'): -9,
		('S', 'O'): -6,
		('O', 'S'): -6,
		('H', 'O'): -6,
		('O', 'H'): -6
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

settings = None
def init(session):
	global settings
	# don't initialize a zillion times, which would also overwrite
	# any changed but not saved settings
	if settings is None:
		settings = _MatchmakerSettings(session, "matchmaker")
