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

def swapaa(session, residues, res_type, *, lib=None, criteria="dchp", preserve=None, retain=False, log=True,
    ignore_other_models=False, density=None, overlap_cutoff=0.6, hbond_allowance=0.4, score_method="num",
    relax=True, dist_slop=None, angle_slop=None):
    '''
    Command to swap amino acid side chains
    '''
    from chimerax.core.errors import UserError
    residues = [r for r in residues if r.polymer_type == r.PT_AMINO]
    if not residues:
        raise UserError("No amino acid residues specified for swapping")

    # res_type and lib are handled by underlying call

    if type(criteria) == str:
        for c in criteria:
            if c not in "dchp":
                raise UserError("Unknown criteria: '%s'" % c)

    from . import swap_res
    try:
        swap_res.swapaa(session, residues, res_type, lib=lib, criteria=criteria, preserve=preserve,
            retain=retain, log=log, ignore_other_models=ignore_other_models, density=density,
            overlap_cutoff=overlap_cutoff, hbond_allowance=hbond_allowance, score_method=score_method,
            relax=relax, dist_slop=dist_slop, angle_slop=angle_slop)
    except swap_res.SwapError as e:
        raise UserError(e)

def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, StringArg, BoolArg, IntArg, Or, FloatArg, EnumOf
    from chimerax.atomic import ResiduesArg
    from chimerax.map import MapArg
    desc = CmdDesc(
        required = [('residues', ResiduesArg), ('res_type', StringArg)],
        keyword = [('lib', StringArg), ('criteria', Or(IntArg, StringArg))), ('preserve', FloatArg),
            ('retain', BoolArg), ('log', BoolArg), ('ignore_other_models', BoolArg), ('density', MapArg),
            ('overlap_cutoff', FloatArg), ('hbond_allowance', FloatArg),
            ('score_method', EnumOf(('sum', 'num')), ('relax', BoolArg), ('dist_slop', FloatArg),
            ('angle_slop', FloatArg)
        ],
        synopsis = 'Swap amino acid side chain(s)'
    )
    register('swapaa', desc, swapaa, logger=logger)
