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

"""
struct_edit: molecular editing utilities
========================================

Utilities routines and constants useful for editing structures.  Higher level than
the basic methods offered by Structure, etc. and therefore less likely to result in
catastrophic error. :-)
"""

# ideal peptide backbone heavy-atom distances
DIST_N_C = 1.335
DIST_CA_N = 1.449
DIST_C_CA = 1.522
DIST_C_O = 1.229

from math import pi, cos, sin

def add_atom(name, element, residue, loc, serial_number=None, bonded_to=None,
                        occupancy=None, info_from=None, alt_loc=None, bfactor=None):
    """Add an atom at the Point 'loc'

       'element' can be a string (atomic symbol), integer (atomic number),
       or an Element instance.

       The atom is added to the given residue (and its molecule).
       'loc' can be an array of xyzs if there are multiple coordinate sets.

       If no 'serial_number' is given, then the atom will be given a serial
       number one greater than the largest serial number of the other atoms
       in the structure.

       'bonded_to' is None or an Atom.  If an Atom, then the new atom
       inherits various attributes [display, altloc, style, occupancy]
       from that atom and a bond to that Atom is created.

       If 'info_from' is supplied then the information normally garnered
       from the 'bonded_to' atom will be obtained from the 'info_from'
       atom instead. Typically used when there is no 'bonded_to' atom.

       If 'occupancy' is not None or the 'bonded_to' atom is not None,
       the new atom will be given the corresponding occupancy.

       If 'bfactor' is not None or the 'bonded_to' atom is not None,
       the new atom will be given the corresponding bfactor.

       If 'alt_loc' is specified (must be a single-character string), then
       the new atom will be given that alt loc, otherwise the alt loc will be ' '.

       Returns the new atom.
    """

    if not info_from:
        info_from = bonded_to
    struct = residue.structure
    new_atom = struct.new_atom(name, element)
    residue.add_atom(new_atom)
    if alt_loc is not None:
        new_atom.set_alt_loc(alt_loc, True)
    from numpy import array
    if len(loc.shape) == 1:
        locs = array([loc])
    else:
        locs = loc
    if struct.num_coordsets == 0:
        if len(locs) > 1:
            from chimerax.core.errors import LimitationError
            raise LimitationError("Cannot add_atom() multi-position atom to empty structure")
        new_atom.coord = locs[0]
    else:
        for xyz, cs_id in zip(locs, struct.coordset_ids):
            new_atom.set_coord(xyz, cs_id)
    if serial_number is None:
        import numpy
        serial_number = numpy.max(struct.atoms.serial_numbers) + 1
    new_atom.serial_number = serial_number
    if occupancy is not None or info_from and hasattr(info_from, 'occupancy'):
        new_atom.occupancy = getattr(info_from, 'occupancy', occupancy)
    if bfactor is not None or info_from and hasattr(info_from, 'bfactor'):
        new_atom.bfactor = getattr(info_from, 'bfactor', bfactor)
    if info_from:
        new_atom.display = info_from.display
        new_atom.draw_mode = info_from.draw_mode
    if bonded_to:
        add_bond(new_atom, bonded_to)
    return new_atom

def add_dihedral_atom(name, element, n1, n2, n3, dist, angle, dihed,
        structure=None, residue=None, bonded=False, occupancy=None,
        info_from=None):
    """Add an atom given 3 Atoms/xyzs and angle/distance constraints
    
       'element' can be a string (atomic symbol), integer (atomic number),
       or an Element instance.

       The atom is added to the given structure.  If no structure or
       residue is specified, then n1/n2/n3 must be Atoms and the new atom
       is added to n1's structure and residue.  If just residue is
       specified, the new atom is added to that residue and its structure.

       'n1' marks the position from which 'dist' is measured, and in
       combination with 'n2' forms 'angle', and then with 'n3' forms
       'dihed'.

       if 'bonded' is True then n1 must be an Atom and the new atom will
       be bonded to it.

       If 'occupancy' is not None or the 'bonded' is True and n1 has an
       occupancy, the new atom will be given the corresponding occupancy.

       if 'info_from' is supplied (needs to be an Atom), miscellaneous
       info (see add_atom() doc string) will be obtained from that atom.

       Returns the new atom.
    """

    if bonded:
        bonded_to = n1
    else:
        bonded_to = None
    from . import Atom
    if n1.__class__ is Atom:
        if not residue:
            structure = n1.structure
            residue = n1.residue
        n1 = n1.coord
        n2 = n2.coord
        n3 = n3.coord
    if not structure:
        structure = residue.structure
    
    final_pt = find_pt(n1, n2, n3, dist, angle, dihed)

    return add_atom(name, element, residue, final_pt, bonded_to=bonded_to, occupancy=occupancy,
        info_from=info_from)

def add_bond(a1, a2, halfbond=None, color=None):
    if a1.num_bonds > 0:
        sample_bond = a1.bonds[0]
    elif a2.num_bonds > 0:
        sample_bond = a2.bonds[0]
    else:
        sample_bond = None
    if halfbond is None:
        if sample_bond:
            halfbond = sample_bond.halfbond
        else:
            halfbond = True
    try:
        b = a1.structure.new_bond(a1, a2)
    except TypeError as e:
        from chimerax.core.errors import UserError
        raise UserError(str(e))
    b.halfbond = halfbond
    if not halfbond:
        if color is None:
            if sample_bond:
                color = sample_bond.color
            else:
                color = a1.color
        b.color = color
    if a1.residue == a2.residue:
        return b

    # this is a cross-residue bond, may need to reorder residues
    is_start = []
    is_end = []
    all_residues = a1.structure.residues
    # order the two residues based on sequence number/insertion code,
    # so that the most "natural" reordering occurs if possible
    r1, r2 = a1.residue, a2.residue
    if r1 < r2:
        residues = (r1, r2)
    else:
        residues = (r2, r1)
    indices = [all_residues.index(r) for r in residues]
    if indices[0]+1 == indices[1] or indices[1]+1 == indices[0]:
        # already adjacent
        return b
    for i, r in zip(indices, residues):
        is_start.append(i == 0 or not r.connects_to(all_residues[i-1]))
        is_end.append(i == len(all_residues)-1 or not r.connects_to(all_residues[i+1]))
    if is_end[0] and is_start[1]:
        if indices[0] < indices[1]:
            # move rear residues forward, closing gap
            close_gap, i1, i2 = True, indices[0], indices[1]
        else:
            # move forward residues back, across rear residues
            close_gap, i1, i2 = False, indices[1], indices[0]
    elif is_start[0] and is_end[1]:
        if indices[0] < indices[1]:
            # move forward residues back, across rear residues
            close_gap, i1, i2 = False, indices[0], indices[1]
        else:
            # move rear residues forward, closing gap
            close_gap, i1, i2 = True, indices[1], indices[0]
    else:
        return b
    def find_end(pos, dir=1):
        def test(pos):
            if dir == 1:
                return pos < len(all_residues) - 1
            return pos > 0
        while test(pos):
            if all_residues[pos].connects_to(all_residues[pos+dir]):
                pos += dir
            else:
                break
        return pos
    if close_gap:
        end_range = find_end(i2)
        new_residues = list(all_residues[0:i1+1]) + list(all_residues[i2:end_range+1]) \
            + list(all_residues[i1+1:i2]) + list(all_residues[end_range+1:])
    else:
        er1 = find_end(i1)
        er2 = find_end(i2, dir=-1)
        new_residues = list(all_residues[0:i1]) + list(all_residues[er1+1:i2+1]) \
            + list(all_residues[i1:er1+1]) + list(all_residues[i2+1:])
    a1.structure.reorder_residues(new_residues)
    return b

def find_pt(n1, n2, n3, dist, angle, dihed):
    # cribbed from Midas addgrp command!
    from numpy.linalg import norm
    from numpy import cross
    normalize = lambda v: v/norm(v)
    v12 = n2 - n1
    v13 = n3 - n1
    v12 = normalize(v12)
    x = normalize(cross(v13, v12))
    y = normalize(cross(v12, x))

    from chimerax.geometry import Place
    xform = Place([(x[i], y[i], v12[i], n1[i]) for i in range(3)])

    rad_angle = pi * angle / 180.0
    tmp = dist * sin(rad_angle)
    rad_dihed = pi * dihed / 180.0
    from numpy import array
    pt = array([tmp*sin(rad_dihed), tmp*cos(rad_dihed), dist*cos(rad_angle)])
    return xform * pt

def gen_atom_name(element, residue):
    """generate non-hydrogen atom name"""
    n = 1
    while True:
        name = "%s%d" % (str(element).upper(), n)
        if not residue.find_atom(name):
            break
        n += 1
    return name

def set_bond_length(bond, bond_length, *, move_smaller_side=True, status=None, undo_state=None):
    bond.structure.idatm_valid = False
    try:
        smaller = bond.smaller_side
    except ValueError:
        if status:
            status("Bond is involved in ring/cycle.\nMoved bonded atoms (only) equally.", color="blue")
        mid = sum([a.coord for a in bond.atoms]) / 2
        factor = bond_length / bond.length
        for a in bond.atoms:
            a.coord = (a.coord - mid) * factor + mid
        return
    bigger = bond.other_atom(smaller)
    if move_smaller_side:
        moving = smaller
        fixed = bigger
    else:
        moving = bigger
        fixed = smaller
    mp = moving.coord
    fp = fixed.coord
    v1 = mp - fp
    from numpy.linalg import norm
    v1_len = norm(v1)
    v1 *= bond_length / v1_len
    delta = v1 - (mp - fp)
    moving_atoms = bond.side_atoms(moving)
    new_coords = moving_atoms.coords + delta
    if undo_state:
        undo_state.add(moving_atoms, "coords", moving_atoms.coords, new_coords)
    moving_atoms.coords = new_coords

standardization_info = { "5BU": "U", "CSL": "C", "MSE": "MET", "UMS": "U" }
standardizable_residues = list(standardization_info.keys())

def standardize_residues(session, residues, *, res_types=standardizable_residues, verbose=True):
    from . import Residues
    if not isinstance(residues, Residues):
        residues = Residues(residues)
    for res_type in res_types:
        target_residues = residues.filter(residues.names == res_type)
        if not target_residues:
            continue
        target_type = standardization_info[res_type]
        results = {}
        exec('func = _standardize_%s' % res_type, globals(), results)
        func = results['func']
        for r in target_residues:
            func(session, r, verbose=verbose)
            r.name = target_type
            if verbose:
                session.logger.info("Residue %s changed %s\N{RIGHTWARDS ARROW}%s"
                    % (r, res_type, target_type))

def _standardize_CSL(session, r, *, verbose=True):
    _mutate_sugar_Se(session, r)

def _standardize_5BU(session, r, *, verbose=True):
    for a in r.atoms:
        if a.element == "Br":
            r.structure.delete_atom(a)
            break

def _standardize_MSE(session, r, *, verbose=True):
    for a in r.atoms:
        if a.element != "Se":
            continue
        a.element = "S"
        a.name = "SD"
        a.idatm_type = "S3"
        for nb, b in zip(a.neighbors, a.bonds):
            if nb.name == "CE":
                set_bond_length(b, 1.78, status=session.logger.status)
            elif nb.name == "CG":
                set_bond_length(b, 1.81, status=session.logger.status)

def _standardize_UMS(session, r, *, verbose=True):
    _mutate_sugar_Se(session, r)

def _mutate_sugar_Se(session, r):
    for a in r.atoms:
        if a.name == "CA'":
            for nb in a.neighbors:
                if nb.element == 1:
                    r.structure.delete_atom(nb)
            r.structure.delete_atom(a)
            break
    for a in r.atoms:
        if a.element != "Se":
            continue
        a.element = "O"
        a.name = "O2'"
        a.idatm_type = "O3"
        for nb, b in zip(a.neighbors, a.bonds):
            if nb.name == "C2'":
                set_bond_length(b, 1.43, status=session.logger.status)
