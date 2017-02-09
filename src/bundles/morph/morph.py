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

#
def morph(session, structures, frames = 20, rate = 'linear', method = 'corkscrew', cartesian = False):
    '''
    Morph between atomic models using Yale Morph Server algorithm.

    Parameters
    ----------
    structures : list of Structure
        Configurations to morph
    frames : int
        Number of frames for each morph segment. One less than this value is new
        configurations are inserted between every pair of structures given.
    rate : 'linear', 'ramp down', 'ramp up' or 'sinusoidal'
        The rate of morphing from one state to the next.
    method : 'corkscrew', 'independent', 'linear'
        How hinged groups of atoms are morphed.
    cartesian : bool
        Whether to interpolate x,y,z atom coordinates or use internal coordinates
        which preserve bond lengths.
    '''

    if len(structures) < 2:
        from chimerax.core.errors import UserError
        raise UserError('Require at least 2 structures for morph')

    from .motion import compute_morph
    traj = compute_morph(structures, session.logger, method=method, rate=rate, frames=frames, cartesian=cartesian)
    session.models.add([traj])

    session.logger.info('Computed %d frame morph #%s' % (traj.num_coord_sets, traj.id_string()))
    from .interp_residue import iit, dpt
    from .interpolate import smt
    print ('interpInternal time', iit, 'dihedral point time', dpt, 'segment motion time', smt)

# -----------------------------------------------------------------------------------------
#
def register_morph_command():
    from chimerax.core.commands import CmdDesc, register, StructuresArg, IntArg, EnumOf, BoolArg
    desc = CmdDesc(
        required = [('structures', StructuresArg)],
        keyword = [('frames', IntArg),
                   ('rate', EnumOf(('linear', 'ramp up', 'ramp down', 'sinusoidal'))),
                   ('method', EnumOf(('corkscrew', 'independent', 'linear'))),
                   ('cartesian', BoolArg)],
        synopsis = 'morph atomic structures'
    )
    register('morph', desc, morph)
