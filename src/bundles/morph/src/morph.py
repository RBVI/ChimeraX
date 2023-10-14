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

#
def morph(session, structures, frames = 50, wrap = False, rate = 'linear', method = 'corkscrew',
          cartesian = False, same = False, core_fraction = 0.5, min_hinge_spacing = 6,
          hide_models = True, play = True, slider = True, color_segments = False, color_core = None):
    '''
    Morph between atomic models using Yale Morph Server algorithm.

    Parameters
    ----------
    structures : list of Structure
        Configurations to morph
    frames : int
        Number of frames for each morph segment. One less than this value is new
        configurations are inserted between every pair of structures given.
    wrap : bool
        If true then morph continues from last structure to first.  This is the same
        as if the first structure is appended to structures to morph.
    rate : 'linear', 'ramp down', 'ramp up' or 'sinusoidal'
        The rate of morphing from one state to the next.
    method : 'corkscrew', 'linear'
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
    slider : bool
        Whether to show a slider to play through the morph.  Default true.
    color_segments : bool
        Whether to color the residues for each rigid segment with a unique color.
        This is to see how the morph algorithm divided the structure into segments.
        For morphing a sequence of 3 or more structures only the residues segments
        for the morph between the first two in the sequence is shown.  Segments are
        recomputed for each consecutive pair in the sequence.  Default false.
    color_core : Color or None
        Color the core residues the specified color.  This is to understand what residues
        the algorithm calculates to be the core.
    '''

    if len(structures) < 2:
        from chimerax.core.errors import UserError
        raise UserError('Require at least 2 structures for morph')

    if wrap:
        structures.append(structures[0])
        
    from .motion import compute_morph
    traj = compute_morph(structures, session.logger, method=method, rate=rate, frames=frames,
                         cartesian=cartesian, match_same=same, core_fraction = core_fraction,
                         min_hinge_spacing = min_hinge_spacing,
                         color_segments = color_segments, color_core = color_core)
    session.models.add([traj])
    if not color_segments and color_core is None:
        if traj.num_chains == 1:
            # Assign new color for single chain morphs for visual clarity
            traj.set_initial_color()

    session.logger.info('Computed %d frame morph #%s' % (traj.num_coordsets, traj.id_string))

    if hide_models:
        for m in structures:
            m.display = False

    if slider and session.ui.is_gui:
        from chimerax.std_commands.coordset import coordset_slider
        coordset_slider(session, [traj])

    if play:
        csids = traj.coordset_ids
        cmd = 'coordset #%s %d,%d' % (traj.id_string, min(csids), max(csids))
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
    from chimerax.core.commands import CmdDesc, register, IntArg, EnumOf, BoolArg, FloatArg, ColorArg
    from chimerax.atomic import StructuresArg
    desc = CmdDesc(
        required = [('structures', StructuresArg)],
        keyword = [('frames', IntArg),
                   ('wrap', BoolArg),
                   ('rate', EnumOf(('linear', 'ramp up', 'ramp down', 'sinusoidal'))),
                   ('method', EnumOf(('corkscrew', 'linear'))),
                   ('cartesian', BoolArg),
                   ('same', BoolArg),
                   ('core_fraction', FloatArg),
                   ('min_hinge_spacing', IntArg),
                   ('hide_models', BoolArg),
                   ('play', BoolArg),
                   ('slider', BoolArg),
                   ('color_segments', BoolArg),
                   ('color_core', ColorArg)],
        synopsis = 'morph atomic structures'
    )
    register('morph', desc, morph, logger=logger)
