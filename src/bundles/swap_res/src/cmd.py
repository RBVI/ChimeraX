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

default_criteria = "dhcp"
def swap_aa(session, residues, res_type, *, angle_slop=None, bfactor=None, criteria=default_criteria,
    density=None, dist_slop=None, hbond_allowance=None, ignore_other_models=False, lib=None, log=True,
    preserve=None, relax=True, retain=False, score_method="num", overlap_cutoff=None):
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

    if lib is None:
        available_libs = session.rotamers.library_names()
        for lib_name in available_libs:
            if "Dunbrack" in lib_name:
                lib = lib_name
                break
        else:
            if available_libs:
                lib = available_libs[0]
            else:
                raise UserError("No rotamer libraries installed!")
    if log:
        session.logger.info("Using %s library" % lib)

    from . import swap_res
    swap_res.swap_aa(session, residues, res_type, bfactor=bfactor, clash_hbond_allowance=hbond_allowance,
        clash_score_method=score_method, clash_overlap_cutoff=overlap_cutoff,
        criteria=criteria, density=density, hbond_angle_slop=angle_slop,
        hbond_dist_slop=dist_slop, ignore_other_models=ignore_other_models, lib=lib, log=log,
        preserve=preserve, hbond_relax=relax, retain=retain)

def register_command(command_name, logger):
    from chimerax.core.commands import CmdDesc, register, StringArg, BoolArg, IntArg, Or, FloatArg, EnumOf
    from chimerax.core.commands import NonNegativeFloatArg, DynamicEnum
    from chimerax.atomic import ResiduesArg
    from chimerax.map import MapArg
    desc = CmdDesc(
        required = [('residues', ResiduesArg), ('res_type', StringArg)],
        keyword = [
            ('angle_slop', FloatArg),
            ('bfactor', FloatArg),
            ('criteria', Or(IntArg, StringArg)),
            ('density', MapArg),
            ('dist_slop', FloatArg),
            ('hbond_allowance', FloatArg),
            ('ignore_other_models', BoolArg),
            ('lib', DynamicEnum(logger.session.rotamers.library_names)),
            ('log', BoolArg),
            ('preserve', NonNegativeFloatArg),
            ('relax', BoolArg),
            ('retain', BoolArg),
            ('score_method', EnumOf(('sum', 'num'))),
            ('overlap_cutoff', FloatArg),
        ],
        synopsis = 'Swap amino acid side chain(s)'
    )
    register(command_name, desc, swap_aa, logger=logger)
