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

def mcopy(session, from_atoms, to_atoms = None, settings = 'csv', match_residues = True):
    '''Copy settings (colors, styles, positions...) from one atomic structure to another.

    Parameters
    ----------
    from_atoms : Atoms
        Atoms to copy attributes from.
    to_atoms : Atoms
        Atoms to copy attributes to.
    settings : string
    	String specifying which attributes to copy.  Default 'csv'.
        'c' = colors, 's' = styles, 'v' = visibility, 'l' = labels,
        'x' = coordinates, 'y' = aligned coordinates, 'p' = position,
        'h' = secondary structure, 'a' = all
    match_residues : bool
        Whether to match atoms with same chain id, residue number, residue name
        and atom name.  Default True.  If false, then residues are matched in the
        order they are listed in the atomic structure.
    '''

    if to_atoms.intersects(from_atoms):
        from chimerax.core.errors import UserError
        raise UserError('mcopy: The two sets of atoms cannot overlap')
    
    ma, upa, mb, upb, mr, upr, mm = match_objects(from_atoms, to_atoms, match_residues)

    ct = (('c', copy_colors),
          ('s', copy_style),
          ('v', copy_visibility),
          ('l', copy_labels),
          ('x', copy_coordinates),
          ('y', copy_aligned_coordinates),
          ('p', copy_placement),
          ('h', copy_secondary_structure),
          )

    settings = settings.lower()
    if 'a' in settings:
        settings = 'csvlxp'

    for c, cf in ct:
        if c in settings:
            cf(ma, mb, mr, mm)

    if upa > 0 or upb > 0 or upr > 0:
        session.logger.status('mcopy: %d unpaired atoms, %d unpaired bonds, %d unpaired residues' % (upa, upb, upr))

# -----------------------------------------------------------------------------
#
def register_mcopy_command(logger):
    from chimerax.core.commands import register, CmdDesc, StringArg, BoolArg
    from chimerax.atomic import AtomsArg
    desc = CmdDesc(required=[('from_atoms', AtomsArg)],
                   keyword=[('to_atoms', AtomsArg),
                            ('settings', StringArg),
                            ('match_residues', BoolArg),
                            ],
                   required_arguments = ['to_atoms'],
                   synopsis='Copy settings from one atomic structure to another')
    register('mcopy', desc, mcopy, logger=logger)

# -----------------------------------------------------------------------------
#
def match_objects(from_atoms, to_atoms, match_residues):

    fms = from_atoms.unique_structures
    if len(fms) != 1:
        from chimerax.core.errors import UserError
        raise UserError('Must specify one source molecule, got %d' % len(fms))
    fm = fms[0]
    tms = to_atoms.unique_structures
    mm = {fm : tms}

    if match_residues:
        def rid(r):
            return (r.chain_id, r.number, r.name)
    else:
        rorder = {}
        for atoms in (from_atoms, to_atoms):
            rn = 0
            for r in atoms.unique_residues:
                if r not in rorder:
                    rorder[r] = rn
                    rn += 1
        def rid(r, rorder=rorder):
            return rorder[r]
        
    mr = {}
    upr = 0
    frt = {}
    for r in from_atoms.unique_residues:
        frt[rid(r)] = r
        mr[r] = []
    for r in to_atoms.unique_residues:
        k = rid(r)
        if k in frt:
            mr[frt[k]].append(r)
        else:
            upr += 1

    ma = {}
    upa = 0
    fat = {}
    for a in from_atoms:
        r = a.residue
        fat[(rid(r), a.name)] = a
        ma[a] = []
    for a in to_atoms:
        r = a.residue
        k = (rid(r), a.name)
        if k in fat:
            ma[fat[k]].append(a)
        else:
            upa += 1

    mb = {}
    upb = 0
    for a in ma.keys():
        for b in a.bonds:
            ao = b.other_atom(a)
            if a < ao and ao in ma:
                tbl = []
                for ta in ma[a]:
                    tbl.extend([tb for tb in ta.bonds if tb.other_atom(ta) in ma[ao]])
                if tbl:
                    mb[b] = tbl
                upb += len(ma[a]) - len(tbl)

    return ma, upa, mb, upb, mr, upr, mm

# -----------------------------------------------------------------------------
#
def copy_colors(ma, mb, mr, mm):

    for a, tal in ma.items():
        for ta in tal:
            ta.color = a.color

    for b, tbl in mb.items():
        for tb in tbl:
            tb.color = b.color
            tb.halfbond = b.halfbond

    for r, trl in mr.items():
        for tr in trl:
            tr.ribbon_color = r.ribbon_color
            tr.ring_color = r.ring_color
            
# -----------------------------------------------------------------------------
#
def copy_style(ma, mb, mr, mm):

    for a, tal in ma.items():
        for ta in tal:
            ta.draw_mode = a.draw_mode
            ta.radius = a.radius

    for b, tbl in mb.items():
        for tb in tbl:
            tb.radius = b.radius

    for r, trl in mr.items():
        for tr in trl:
            tr.thin_rings = r.thin_rings

    for m, tml in mm.items():
        for tm in tml:
            tm.autochain = m.autochain
            tm.ball_scale = m.ball_scale
            tm.bond_radius = m.bond_radius
            tm.pseudobond_radius = m.pseudobond_radius
            tm.ribbon_tether_scale = m.ribbon_tether_scale
            tm.ribbon_tether_shape = m.ribbon_tether_shape
            tm.ribbon_tether_sides = m.ribbon_tether_sides
            tm.ribbon_tether_opacity = m.ribbon_tether_opacity
            tm.ribbon_mode_helix = m.ribbon_mode_helix
            tm.ribbon_mode_strand = m.ribbon_mode_strand
            tm.ribbon_orientation = m.ribbon_orientation
            copy_ribbon_xsection(m.ribbon_xs_mgr, tm.ribbon_xs_mgr)

# -----------------------------------------------------------------------------
#
def copy_ribbon_xsection(from_xsection, to_xsection):

    attrs = [
        'scale_helix',
        'scale_helix_arrow',
        'scale_sheet',
        'scale_sheet_arrow',
        'scale_coil',
        'scale_nucleic',
        'tube_radius',
        'style_helix',
        'style_sheet',
        'style_coil',
        'style_nucleic',
        'arrow_helix',
        'arrow_sheet',
        'params',
        'transitions',
        ]
    for attr in attrs:
        setattr(to_xsection, attr, getattr(from_xsection, attr))
    to_xsection._set_gc_ribbon()

# -----------------------------------------------------------------------------
#
def copy_visibility(ma, mb, mr, mm):

    for a, tal in ma.items():
        for ta in tal:
            ta.display = a.display
            ta.hide = a.hide

    for b, tbl in mb.items():
        for tb in tbl:
            tb.display = b.display

    for r, trl in mr.items():
        for tr in trl:
            tr.ribbon_display = r.ribbon_display
            tr.ribbon_hide_backbone = r.ribbon_hide_backbone
            tr.ring_display = r.ring_display

    for m, tml in mm.items():
        for tm in tml:
            tm.display = m.display
            
# -----------------------------------------------------------------------------
#
def copy_labels(ma, mb, mr, mm):

    o2l = {}  # Map atoms,bonds,residues to ObjectLabel
    from chimerax.label.label3d import ObjectLabels, label as create_label
    session = tuple(mm.keys())[0].session
    for lm in session.models.list(type = ObjectLabels):
        for label in lm.labels():
            o2l[label.object] = label

    from chimerax.core.objects import Objects
    from chimerax.atomic import Atoms, Bonds
    
    for a, tal in ma.items():
        label = o2l.get(a)
        if label:
            for ta in tal:
                tao = Objects(atoms = Atoms([ta]))
                create_label(session, objects = tao, object_type = 'atoms',
                             **label_attributes(label))

    for b, tbl in mb.items():
        label = o2l.get(b)
        if label:
            for tb in tbl:
                tbo = Objects(bonds = Bonds([tb]))
                create_label(session, objects = tbo, object_type = 'bonds',
                             **label_attributes(label))

    for r, trl in mr.items():
        label = o2l.get(r)
        if label:
            for tr in trl:
                tro = Objects(atoms = tr.atoms)
                create_label(session, objects = tro, object_type = 'residues',
                             **label_attributes(label))

# -----------------------------------------------------------------------------
#
def label_attributes(label):

    attrs = {'offset': label.offset,
            'color': label.color,
            'bg_color': label.background,
            'size': label.size,
            'height': label.height,
            'font': label.font}
    if label.attribute:
        attrs['attribute'] = label.attribute
    else:
        attrs['text'] = label.text
    return attrs

# -----------------------------------------------------------------------------
#
def copy_coordinates(ma, mb, mr, mm):

    for a, tal in ma.items():
        for ta in tal:
            ta.coord = a.coord

# -----------------------------------------------------------------------------
#
def copy_aligned_coordinates(ma, mb, mr, mm):

    for a, tal in ma.items():
        for ta in tal:
            ta.scene_coord = a.scene_coord

# -----------------------------------------------------------------------------
#
def copy_placement(ma, mb, mr, mm):

    for m, tml in mm.items():
        for tm in tml:
            tm.scene_position = m.scene_position

# -----------------------------------------------------------------------------
#
def copy_secondary_structure(ma, mb, mr, mm):

    for r, trl in mr.items():
        for tr in trl:
            tr.ss_type = r.ss_type
