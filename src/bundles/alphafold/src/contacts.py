# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This particular file is part of the ChimeraX library. You can also
# redistribute and/or modify it under the terms of the GNU Lesser General
# Public License version 2.1 as published by the Free Software Foundation.
# For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
# EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
# LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
# VERSION 2.1
#
# This notice must be embedded in or attached to all copies, including partial
# copies, of the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===



# TODO: Add option to show all intra-chain contacts




# -----------------------------------------------------------------------------
#
def alphafold_contacts(session, atoms, to_atoms = None, distance = 3, max_pae = None,
                       flip = False, palette = None, range = None, radius = 0.2, dashes = 1,
                       name = 'PAE Contacts', replace = True, output_file = None,
                       method = 'alphafold'):
    '''
    Create pseudobonds between close residues or atoms of an AlphaFold structure
    colored by the predicted aligned error value.  The paecontacts colormap
    is used ranging from blue for low error to red for high error.
    Pseudobonds are drawn between CA atoms.
    '''
    
    # Get structure containing residues.
    if len(atoms) == 0:
        from chimerax.core.errors import UserError        
        raise UserError(f'No alphafold specified for {method} contacts')
    s = atoms[0].structure

    # If to_residues not specified use all other residues in same structure.
    if to_atoms is None:
        to_atoms = s.atoms - atoms
        if len(to_atoms) == 0:
            to_atoms = atoms

    # Check that we have some residues
    if len(to_atoms) == 0:
        from chimerax.core.errors import UserError        
        raise UserError(f'No to atoms specified for {method} contacts')

    # Make sure all residues belong to one structure
    ns = len((atoms | to_atoms).unique_structures)
    if ns > 1:
        from chimerax.core.errors import UserError        
        raise UserError('Interface PAE pseudobonds can only be computed for a single structure, got %d.' % ns)

    # Make sure structure has PAE data opened.
    pae = getattr(s, f'{method}_pae', None)
    if pae is None:
        from chimerax.core.errors import UserError
        raise UserError('Structure %s does not have PAE data opened' % s)

    # Use pae palette if none specified.
    if palette is None:
        from chimerax.core.colors import BuiltinColormaps
        palette = BuiltinColormaps['paecontacts']

    # Adjust palette range
    from chimerax.core.colors import colormap_with_range
    palette = colormap_with_range(palette, range, full_range = (0,30))

    g = s.pseudobond_group(name)
    if replace:
        # Delete old pseudobonds
        g.pseudobonds.delete()
    g.dashes = dashes

    # Replace protein or nucleic atoms with residues
    res, atoms = _pae_residues_and_atoms(atoms)
    to_res, to_atoms = _pae_residues_and_atoms(to_atoms)
    
    # Get pairs of close residues
    close_rapairs = (_close_residue_pairs(res, to_res, distance) +
                     _close_residue_atom_pairs(res, to_atoms, distance) +
                     _close_atom_residue_pairs(atoms, to_res, distance) +
                     _close_atom_pairs(atoms, to_atoms, distance))
    rapairs = [(ra1,ra2) for ra1, ra2 in close_rapairs if ra1 != ra2]

    # Get pae values
    if flip:
        pae_values = [pae.value(ra2,ra1) for ra1, ra2 in rapairs]
    else:
        pae_values = [pae.value(ra1,ra2) for ra1, ra2 in rapairs]

    # Show only contacts below max pae value.
    if max_pae is not None:
        rapairs = [rapairs[i] for i,pae in enumerate(pae_values) if pae <= max_pae]
        pae_values = [pae for pae in pae_values if pae <= max_pae]
        
    # Create pseudobonds between close residues
    pblist = []
    from chimerax.atomic import Residue
    for ra1, ra2 in rapairs:
        a1 = ra1.principal_atom if isinstance(ra1, Residue) else ra1
        a2 = ra2.principal_atom if isinstance(ra2, Residue) else ra2
        if a1 is None or a2 is None:
            continue	# TODO: Warn about missing principal atoms
        b = g.new_pseudobond(a1, a2)
        b.radius = radius
        pblist.append(b)
    from chimerax.atomic import Pseudobonds
    pbonds = Pseudobonds(pblist)

    # Color pseudobonds
    pbonds.colors = palette.interpolated_rgba8(pae_values)

    if output_file is not None:
        lines = [_residue_pair_line(ra1, ra2, pae_value)
                 for (ra1,ra2),pae_value in zip (rapairs, pae_values)]
        with open(output_file, 'w') as f:
            f.write('\n'.join(lines))

    msg = f'Found {len(rapairs)} residue or atom pairs within distance %.3g' % distance
    if max_pae is not None:
        msg += ' with pae <= %.3g' % max_pae
    session.logger.status(msg, log=True)

    return pbonds

# -----------------------------------------------------------------------------
#
def _pae_residues_and_atoms(atoms):
    res = set()
    np_atoms = []
    from .pae import per_residue_pae
    for a in atoms:
        r = a.residue
        if per_residue_pae(r):
            res.add(r)
        else:
            np_atoms.append(a)
    from chimerax.atomic import Residues, Atoms
    return Residues(tuple(res)), Atoms(np_atoms)

# -----------------------------------------------------------------------------
#
def _close_residue_pairs(residues1, residues2, distance):
    '''Find pairs of residues within specified distance of each other.'''
    rpairs = []
    atoms2 = residues2.atoms
    xyz2 = atoms2.scene_coords
    from chimerax.geometry import find_close_points
    for r1 in residues1:
        atoms1 = r1.atoms
        xyz1 = atoms1.scene_coords
        i1, i2 = find_close_points(xyz1, xyz2, distance)
        if len(i2) > 0:
            rclose = atoms2[i2].unique_residues
            rpairs.extend([(r1,rc) for rc in rclose if rc is not r1])
    return rpairs

# -----------------------------------------------------------------------------
#
def _close_residue_atom_pairs(residues, atoms, distance):
    rapairs = []
    axyz = atoms.scene_coords
    from chimerax.geometry import find_close_points
    for r in residues:
        rxyz = r.atoms.scene_coords
        i1, i2 = find_close_points(rxyz, axyz, distance)
        if len(i2) > 0:
            rapairs.extend([(r,a) for a in atoms[i2]])
    return rapairs

# -----------------------------------------------------------------------------
#
def _close_atom_residue_pairs(atoms, residues, distance):
    return [(a,r) for r,a in _close_residue_atom_pairs(residues, atoms, distance)]

# -----------------------------------------------------------------------------
#
def _close_atom_pairs(atoms1, atoms2, distance):
    if len(atoms1) == 0 or len(atoms2) == 0:
        return []

    # First find close atoms to speed up computation
    from chimerax.geometry import find_close_points
    xyz1, xyz2 = atoms1.scene_coords, atoms2.scene_coords
    i1, i2 = find_close_points(xyz1, xyz2, distance)
    if len(i1) == 0 or len(i2) == 0:
        return []

    # Now find every close pair.
    apairs = []
    xyz3 = xyz2[i2]
    for i in i1:
        j, i3 = find_close_points(xyz1[i:i+1], xyz3, distance)
        if len(i3) > 0:
            a1 = atoms1[i]
            apairs.extend([(a1, atoms2[i2[k]]) for k in i3])
        
    return apairs

# -----------------------------------------------------------------------------
#
def _residue_to_atom_pairs(rpairs):
    rapairs = []
    for r1,r2 in rpairs:
        rapairs.extend([(o1,o2)
                        for o1 in _residue_or_non_polymer_atoms(r1)
                        for o2 in _residue_or_non_polymer_atoms(r2)])
    return rapairs

# -----------------------------------------------------------------------------
#
def _residue_or_non_polymer_atoms(residue):
    from .pae import per_residue_pae
    if per_residue_pae(residue):
        return [residue]
    else:
        return residue.atoms

# -----------------------------------------------------------------------------
#
def _residue_pair_line(ra1, ra2, pae_value):
    spec1, spec2 = _residue_or_atom_spec(ra1), _residue_or_atom_spec(ra2)
    return f'{spec1} {spec2} %.2f' % pae_value

# -----------------------------------------------------------------------------
#
def _residue_or_atom_spec(residue_or_atom):
    from chimerax.atomic import Residue
    if isinstance(residue_or_atom, Residue):
        r = residue_or_atom
        return f'/{r.chain_id}:{r.number}'
    else:
        a = residue_or_atom
        r = a.residue
        return f'/{r.chain_id}:{r.number}@{a.name}'

# -----------------------------------------------------------------------------
#
def contacts_command_description():
    from chimerax.core.commands import CmdDesc, register, FloatArg, IntArg, BoolArg, StringArg
    from chimerax.core.commands import ColormapArg, ColormapRangeArg, SaveFileNameArg
    from chimerax.atomic import AtomsArg
    desc = CmdDesc(
        required = [('atoms', AtomsArg)],
        keyword = [('to_atoms', AtomsArg),
                   ('distance', FloatArg),
                   ('max_pae', FloatArg),
                   ('flip', BoolArg),
                   ('palette', ColormapArg),
                   ('range', ColormapRangeArg),
                   ('radius', FloatArg),
                   ('dashes', IntArg),
                   ('name', StringArg),
                   ('replace', BoolArg),
                   ('output_file', SaveFileNameArg)],
        synopsis = 'Make pseudobonds colored by PAE for close residues or atoms'
    )
    return desc
    
# -----------------------------------------------------------------------------
#
def register_alphafold_contacts_command(logger):
    desc = contacts_command_description()
    from chimerax.core.commands import register
    register('alphafold contacts', desc, alphafold_contacts, logger=logger)
