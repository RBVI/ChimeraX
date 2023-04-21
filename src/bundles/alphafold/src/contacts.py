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

# -----------------------------------------------------------------------------
#
def alphafold_contacts(session, residues, to_residues = None, distance = 3,
                       flip = False, palette = None, range = None, radius = 0.2, dashes = 1,
                       name = 'PAE Contacts', replace = True, output_file = None,
                       method = 'alphafold'):
    '''
    Create pseudobonds between close residues of an AlphaFold structure
    colored by the predicted aligned error value.  The paecontacts colormap
    is used ranging from blue for low error to red for high error.
    Pseudobonds are drawn between CA atoms.
    '''
    
    # Get structure containing residues.
    if len(residues) == 0:
        from chimerax.core.errors import UserError        
        raise UserError(f'No residues specified for {method} contacts')
    s = residues[0].structure

    # If to_residues not specified use all other residues in same structure.
    if to_residues is None:
        to_residues = s.residues - residues
        if len(to_residues) == 0:
            to_residues = residues

    # Check that we have some residues
    if len(to_residues) == 0:
        from chimerax.core.errors import UserError        
        raise UserError(f'No to residues specified for {method} contacts')

    # Make sure all residues belong to one structure
    ns = len((residues | to_residues).unique_structures)
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
    if range is not None and range != 'full':
        palette = palette.rescale_range(range[0], range[1], full = True)

    g = s.pseudobond_group(name)
    if replace:
        # Delete old pseudobonds
        g.pseudobonds.delete()
    g.dashes = dashes
        
    # Create pseudobonds between close residues
    rpairs = _close_residue_pairs(residues, to_residues, distance)
    pblist = []
    for r1, r2 in rpairs:
        a1, a2 = r1.find_atom('CA'), r2.find_atom('CA')
        b = g.new_pseudobond(a1, a2)
        b.radius = radius
        pblist.append(b)
    from chimerax.atomic import Pseudobonds
    pbonds = Pseudobonds(pblist)

    # Color pseudobonds
    if flip:
        pae_values = [pae.value(r2,r1) for r1, r2 in rpairs]
    else:
        pae_values = [pae.value(r1,r2) for r1, r2 in rpairs]
    pbonds.colors = palette.interpolated_rgba8(pae_values)

    if output_file is not None:
        lines = [f'/{r1.chain_id}:{r1.number} /{r2.chain_id}:{r2.number} %.2f' % pae_value
                 for (r1,r2),pae_value in zip (rpairs, pae_values)]
        with open(output_file, 'w') as f:
            f.write('\n'.join(lines))
            
    return pbonds

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
def contacts_command_description():
    from chimerax.core.commands import CmdDesc, register, FloatArg, IntArg, BoolArg, StringArg
    from chimerax.core.commands import ColormapArg, ColormapRangeArg, SaveFileNameArg
    from chimerax.atomic import ResiduesArg
    desc = CmdDesc(
        required = [('residues', ResiduesArg)],
        keyword = [('to_residues', ResiduesArg),
                   ('distance', FloatArg),
                   ('flip', BoolArg),
                   ('palette', ColormapArg),
                   ('range', ColormapRangeArg),
                   ('radius', FloatArg),
                   ('dashes', IntArg),
                   ('name', StringArg),
                   ('replace', BoolArg),
                   ('output_file', SaveFileNameArg)],
        synopsis = 'Make pseudobonds colored by PAE for close residues'
    )
    return desc
    
# -----------------------------------------------------------------------------
#
def register_alphafold_contacts_command(logger):
    desc = contacts_command_description()
    from chimerax.core.commands import register
    register('alphafold contacts', desc, alphafold_contacts, logger=logger)
