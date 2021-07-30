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
from chimerax.core.errors import UserError

def make_alignment(session, chains, *, circular=defaults['circular'],
        column_criteria=defaults["column_criteria"], dist_cutoff=defaults["dist_cutoff"],
        gap_char=defaults["gap_char"], iteration_limit=defaults['iteration_limit'],
        min_stretch=defaults['min_stretch'], ref_chain=None, show_alignment=True):
    if len(chains) < 2:
        raise UserError("Must specifiy at least two chains as basis for alignment")
    if ref_chain is None:
        ref_chain = chains[0]
    elif ref_chain not in chains:
        raise UserError("Reference chain must be involved in alignment")
    if len(chains.structures.unique()) != len(chains):
        raise UserError("Specify only one chain per model")

    col_all = column_criteria == "all"
    from ._msa3d import match_to_align
    seqs = match_to_align(list(chains.pointers), dist_cutoff, col_all, gap_char, circular)
    #TODO: lots

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
