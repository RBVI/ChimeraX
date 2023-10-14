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


# -----------------------------------------------------------------------------
#
def rna_path(session, pairs, length = None,
             pattern = 'circle', loop_pattern = 'helix', 
             marker_radius = 2,
             loop_color = (102,154,230,255), stem_color = (255,255,0,255),
             loop_twist = 0, branch_tilt = 0,
             helix_radius = 300, helix_rise = 50,
             sphere_radius = None, sphere_turns = None, sphere_turn_spacing = 60,
             helix_loop_size = 8, helix_loop_rise = 20,
             horseshoe_curve_size = 8, horseshoe_side_size = 10, horseshoe_spacing = 1,
             name = 'RNA path'):

    plist = parse_pairs(pairs)
    from .rna_layout import pair_map, rna_path, color_path, LayoutParameters
    pair_map = pair_map(plist)

    if length is None:
        length = max(pair_map.keys())

    param_names = ['loop_pattern', 'loop_twist', 'branch_tilt',
                   'helix_radius', 'helix_rise',
                   'sphere_radius', 'sphere_turns', 'sphere_turn_spacing',
                   'helix_loop_size', 'helix_loop_rise',
                   'horseshoe_curve_size', 'horseshoe_side_size', 'horseshoe_spacing']
    pvalues = locals()
    params = {pname:pvalues[pname] for pname in param_names}
    layout_parameters = LayoutParameters(**params)

    mset, coords = rna_path(session, length, pair_map,
                            pattern = pattern,
                            marker_radius = marker_radius,
                            layout_parameters = layout_parameters,
                            name = name)
    color_path(mset.atoms, pair_map, loop_color, stem_color)
    session.models.add([mset])
    
    return mset, coords

# -----------------------------------------------------------------------------
#
def parse_pairs(pairs):

    import os.path
    path = os.path.expanduser(pairs)
    if os.path.exists(path):
        from .rna_layout import read_base_pairs
        plist = read_base_pairs(path)
        return plist

    plist = parse_pairs_string(pairs)
    if plist is None:
        from chimerax.core.errors import UserError
        raise UserError('Pairs must be a file or start,end,length '
                        'triples, got "%s"' % pairs)
    return plist

# -----------------------------------------------------------------------------
# Can specify base pairs as comma separate list of start,end,length triples.
#
def parse_pairs_string(pairs):

    try:
        p = [int(i) for i in pairs.split(',')]
    except ValueError:
        return None
    if len(p) < 3 or len(p) % 3 != 0:
        return None
    p3 = zip(p[::3], p[1::3], p[2::3])
    return p3

# -----------------------------------------------------------------------------
#
def rna_model(session, sequence, path = None, start_sequence = 1,
              length = None, pairs = None,
              pattern = 'line', loop_pattern = 'helix',
              loop_color = (102,154,230,255), stem_color = (255,255,0,255),
              p_color = (255,165,0,255),
              loop_twist = 0, branch_tilt = 0,
              helix_radius = 300, helix_rise = 50,
              sphere_radius = None, sphere_turns = None, sphere_turn_spacing = 60,
              helix_loop_size = 8, helix_loop_rise = 20,
              horseshoe_curve_size = 8, horseshoe_side_size = 10, horseshoe_spacing = 1,
              name = 'RNA'):

    from . import rna_layout as RL
    import os.path
    seq_path = os.path.expanduser(sequence)
    if os.path.exists(seq_path):
        seq = RL.read_fasta(seq_path)
    else:
        if set(sequence) - set(('A','C','G','U','T','a','c','g','u','t')):
            from chimerax.core.errors import UserError
            raise UserError('Sequence "%s" does not specify a file and the string contains characters besides A,C,G,U,T' % sequence)
        seq = sequence
    seq = seq[start_sequence-1:]
    if len(seq) == 0:
        from chimerax.core.errors import UserError
        raise UserError('Sequence "%s" is empty' % sequence)

    param_names = ['loop_pattern', 'loop_twist', 'branch_tilt',
                   'helix_radius', 'helix_rise',
                   'sphere_radius', 'sphere_turns', 'sphere_turn_spacing',
                   'helix_loop_size', 'helix_loop_rise',
                   'horseshoe_curve_size', 'horseshoe_side_size', 'horseshoe_spacing']
    pvalues = locals()
    params = {pname:pvalues[pname] for pname in param_names}
    layout_parameters = RL.LayoutParameters(**params)

    if path is None:
        if pairs is None:
            from chimerax.core.errors import UserError
            raise UserError('Must specify either a path or base pairing')
        if length is None:
            length = len(seq)
        plist = parse_pairs(pairs)
        pair_map = RL.pair_map(plist)
        base_placements = RL.rna_path(session, length, pair_map,
                                      pattern = pattern,
                                      layout_parameters = layout_parameters,
                                      name = None)
    else:
        mpath = [m for m in atoms_to_markers(path)
                 if hasattr(m, 'extra_attributes')
                 and 'base_placement' in m.extra_attributes]
        if len(mpath) == 0:
            from chimerax.core.errors import UserError
            raise UserError('No path markers found.')
        base_placements = dict((m.residue.number, m.extra_attributes['base_placement'])
                               for m in mpath)
        pair_map = dict((m.residue.number, m.extra_attributes['paired_with'])
                        for m in mpath if 'paired_with' in m.extra_attributes)

    mol = RL.rna_atomic_model(session, seq, base_placements, name)
    RL.color_stems_and_loops(mol, pair_map, loop_color, stem_color, p_color)

    return mol

# -----------------------------------------------------------------------------
#
def atoms_to_markers(atoms):
    return atoms

# -----------------------------------------------------------------------------
#
def rna_minimize(session, molecule, chunk_size = 10, steps = 100,
                 conjugate_gradient_steps = 100, update_interval = 10,
                 nogui = True):

    from .rna_layout import minimize_rna_backbone
    minimize_rna_backbone(molecule, chunk_size, steps,
                          conjugate_gradient_steps, update_interval, nogui)

# -----------------------------------------------------------------------------
#
def rna_smooth(session, path, radius = 50, spacing = 3.33,
               kink_interval = None, kink_radius = None,
               name = 'smooth path'):

    markers = atoms_to_markers(path)

    from .duplex import smooth_path
    mset = smooth_path(markers, radius, spacing,
                       kink_interval, kink_radius, name)
    return mset

# -----------------------------------------------------------------------------
#
def rna_duplex(session, path, sequence, start_sequence = 1, type = 'DNA'):

    from numpy import array
    path_xyz = array([atom.coord().data() for atom in path])

    import os.path
    from .rna_layout import read_fasta
    seq = read_fasta(sequence) if os.path.exists(sequence) else sequence
    seq = seq[start_sequence-1:]

    from .duplex import make_dna_following_path
    mol = make_dna_following_path(path_xyz, seq, polymer_type = type)

    return mol

# -------------------------------------------------------------------------------------
#
def register_rna_layout_command(logger):
    from chimerax.core.commands import CmdDesc, register, EnumOf, Color8Arg, BoolArg, StringArg
    from chimerax.core.commands import IntArg, PositiveIntArg, FloatArg
    from chimerax.atomic import AtomsArg, StructureArg

    PatternArg = EnumOf(('line', 'circle', 'helix', 'sphere'))
    LoopPatternArg = EnumOf(('helix', 'horseshoe'))

    path_opts = [('pattern', PatternArg),
                 ('loop_pattern', LoopPatternArg),
                 ('marker_radius', FloatArg),
                 ('loop_color', Color8Arg),
                 ('stem_color', Color8Arg),
                 ('loop_twist', FloatArg),
                 ('branch_tilt', FloatArg),
                 ('helix_radius', FloatArg),
                 ('helix_rise', FloatArg),
                 ('sphere_radius', FloatArg),
                 ('sphere_turns', FloatArg),
                 ('sphere_turn_spacing', FloatArg),
                 ('helix_loop_size', IntArg),
                 ('helix_loop_rise', FloatArg),
                 ('horseshoe_curve_size', IntArg),
                 ('horseshoe_side_size', IntArg),
                 ('horseshoe_spacing', IntArg),
                 ('name', StringArg)
    ]

    # Make RNA marker model
    path_desc = CmdDesc(required = [('pairs', StringArg)],
                        optional = [('length', PositiveIntArg)],
                        keyword = path_opts,
                        synopsis = 'create an RNA marker model')
    register('rna path', path_desc, rna_path, logger=logger)

    # Make RNA atomic model
    model_desc = CmdDesc(required = [('sequence', StringArg)],
                         optional = [('path', AtomsArg)],
                         keyword = [('start_sequence', IntArg),
                                    ('length', PositiveIntArg),
                                    ('pairs', StringArg),
                                    ('p_color', Color8Arg)] + path_opts,
                        synopsis = 'create an RNA atomic model')
    register('rna model', model_desc, rna_model, logger=logger)

    '''
    Unported commands.

    # Energy minimize RNA atomic model
    minimize_desc = CmdDesc(required = [('molecule', StructureArg)],
                            keyword = [('chunk_ize', PositiveIntArg),
                                       ('steps', PositiveIntArg),
                                       ('conjugate_gradient_steps', PositiveIntArg),
                                       ('update_interval', PositiveIntArg),
                                       ('nogui', BoolArg)],
                            synopsis = 'energy minimize an RNA atomic model')
    register('rna minimize', minimize_desc, rna_minimize, logger=logger)

    # Smooth an RNA marker model path
    smooth_desc = CmdDesc(required = [('path', AtomsArg)],
                          optional = [('radius', FloatArg),
                                      ('spacing', FloatArg)],
                          keyword = [('kink_interval', IntArg),
                                     ('kink_radius', FloatArg),
                                     ('name', StringArg)],
                          synopsis = 'smooth an RNA marker model path')
    register('rna smooth', smooth_desc, rna_smooth, logger=logger)

    # Make a duplex DNA or RNA atomic model
    duplex_desc = CmdDesc(required = [('path', AtomsArg),
                                      ('sequence', StringArg)],
                          keyword = [('start_sequence', IntArg),
                                     ('type', EnumOf(('RNA', 'DNA')))],
                          synopsis = 'make a duplex DNA or RNA atomic model')
    register('rna duplex', duplex_desc, rna_duplex, logger=logger)

    '''
