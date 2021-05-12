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

from .settings import defaults

def make_alignment(chains, dist_cutoff=defaults["dist_cutoff"], column_criteria=defaults["column_criteria"],
        gap_char=defaults["gap_char"], circular=defaults['circular'], min_stretch=defaults['min_stretch'],
        iteration_limit=defaults['iteration_limit'], show_alignment=True, ref_chain=None):
    pass

def register_command(cmd_name, logger):
    from chimerax.core.commands import CmdDesc, register, NonNegativeFloatArg, EnumOf, CharacterArg, \
        BoolArg, Or, NoneArg, NonNegativeIntArg, PositiveIntArg
    from chimerax.atomic import UniqueChainsArg, ChainArg
    desc = CmdDesc(
        required = [('chains', UniqueChainsArg)],
        keyword = [
            ('circular', BoolArg),
            ('column_criteria', EnumOf(['any', 'all'])),
            ('dist_cutoff', NonNegativeFloatArg),
            ('gap_char', CharacterArg),
            ('iteration_limit', Or(NonNegativeIntArg, NoneArg)),
            ('min_stretch', PositiveIntArg),
            ('ref_chain', ChainArg),
            ('show_alignment', BoolArg),
        ],
        synopsis = 'Create sequence alignment from structural superposition'
    )
    register(cmd_name, desc, make_alignment, logger=logger)
