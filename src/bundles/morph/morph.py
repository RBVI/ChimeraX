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
def morph(session, structures, frames = 20, rate = 'linear', method = 'corkscrew',
          cartesian = False, same = False, core_fraction = 0.5, min_hinge_spacing = 6,
          hide_models = True, play = True, color_segments = False):
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
    same : bool
        Whether to match atoms with same chain id, same residue number and same
        atom name.  Default false.
    core_fraction : float
        Fraction of residues of each chain that align best used to define core and
        non-core residues which are then split into contiguous residues stretches
        where the chain crosses between the two residue sets.  Default 0.5.
    min_hinge_spacing : int
        Minimum number of consecutive residues when splitting chains into rigidly
        moving segments at boundaries between core and non-core residues.  Default 6.
    hide_models : bool
        Whether to hide the input models after morph model is created.  Default true.
    play : bool
        Whether to play the morph.  Default true.
    color_segments : bool
        Whether to color the residues for each rigid segment with a unique color.
        This is to see how the morph algorithm divided the structure into segments.
        For morphing a sequence of 3 or more structures only the residues segments
        for the morph between the first two in the sequence is shown.  Segments are
        recomputed for each consecutive pair in the sequence.  Default false.
    '''

    if len(structures) < 2:
        from chimerax.core.errors import UserError
        raise UserError('Require at least 2 structures for morph')

    from .motion import compute_morph
    traj = compute_morph(structures, session.logger, method=method, rate=rate, frames=frames,
                         cartesian=cartesian, match_same=same, core_fraction = core_fraction,
                         min_hinge_spacing = min_hinge_spacing, color_segments = color_segments)
    session.models.add([traj])
    if not color_segments:
        traj.set_initial_color()

    session.logger.info('Computed %d frame morph #%s' % (traj.num_coord_sets, traj.id_string()))

    if hide_models:
        for m in structures:
            m.display = False

    if play:
        csids = traj.coordset_ids
        cmd = 'coordset #%s %d,%d' % (traj.id_string(), min(csids), max(csids))
        from chimerax.core.commands import run
        run(session, cmd)

    # from .interpolate import smt, stt, rit, rst, rsit
    # from .sieve_fit import svt
    # from .motion import ht,it
    # from .segment import ssvt, satt

    # print('interpolate time', it)
    # print('calc segment transform time', stt)
    # print('make residue interpolators time', rsit)
    # print('rigid interp time', rit)
    # print('residue interp time', rst)
    # print('segment atom move time', smt)

    # print('hinge time', ht)
    # print('shared atoms time', satt)
    # print('sieve time', svt)
    # print('segment sieve time', ssvt)

# -----------------------------------------------------------------------------------------
#
def register_morph_command(logger):
    from chimerax.core.commands import CmdDesc, register, StructuresArg, IntArg, EnumOf, BoolArg, FloatArg
    desc = CmdDesc(
        required = [('structures', StructuresArg)],
        keyword = [('frames', IntArg),
                   ('rate', EnumOf(('linear', 'ramp up', 'ramp down', 'sinusoidal'))),
                   ('method', EnumOf(('corkscrew', 'independent', 'linear'))),
                   ('cartesian', BoolArg),
                   ('same', BoolArg),
                   ('core_fraction', FloatArg),
                   ('min_hinge_spacing', IntArg),
                   ('hide_models', BoolArg),
                   ('play', BoolArg),
                   ('color_segments', BoolArg)],
        synopsis = 'morph atomic structures'
    )
    register('morph', desc, morph, logger=logger)
