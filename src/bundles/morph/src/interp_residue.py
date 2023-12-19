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

from time import time
iit = dpt = 0

def interpolate_linear(indices, coords0, coords1, f, coord_set):
        "Interpolate Cartesian position between start and end position."
        t0 = time()
        for i0 in indices:
                c0 = coords0[i0]
                c1 = coords1[i0]
                c = c0 + (c1 - c0) * f
                coord_set[i0,:] = c
        t1 = time()
        global iit
        iit += t1-t0

def interpolate_dihedrals(indices, coords0, coords1, f, coord_set):
        """
        Computer coordinate of atom a0 by interpolating dihedral angle
        defined by atoms (a0, a1, a2, a3).  Every 4 atom indices in indices
        array are used.
        """
        for i in range(0, len(indices), 4):
                i0,i1,i2,i3 = indices[i:i+4]
                interpolate_dihedral(i0, i1, i2, i3, coords0, coords1, f, coord_set)

# C++ optimized versions for interpolating.
from .morph_cpp import interpolate_linear, interpolate_dihedrals

def interpolate_dihedral(i0, i1, i2, i3, coords0, coords1, f, coord_set):
        """
        Computer coordinate of atom a0 by interpolating dihedral angle
        defined by atoms (a0, a1, a2, a3).
        """
        t0 = time()
        from chimerax.geometry import distance, angle, dihedral, dihedral_point
        c00 = coords0[i0]
        c01 = coords0[i1]
        c02 = coords0[i2]
        c03 = coords0[i3]
        length0 = distance(c00, c01)
        angle0 = angle(c00, c01, c02)
        dihed0 = dihedral(c00, c01, c02, c03)
        c10 = coords1[i0]
        c11 = coords1[i1]
        c12 = coords1[i2]
        c13 = coords1[i3]
        length1 = distance(c10, c11)
        angle1 = angle(c10, c11, c12)
        dihed1 = dihedral(c10, c11, c12, c13)
        length = length0 + (length1 - length0) * f
        angle = angle0 + (angle1 - angle0) * f
        ddihed = dihed1 - dihed0
        if ddihed > 180:
                ddihed -= 360
        elif ddihed < -180:
                ddihed += 360
        dihed = dihed0 + ddihed * f
        c1 = coord_set[i1,:]
        c2 = coord_set[i2,:]
        c3 = coord_set[i3,:]
        t2 = time()
        c0 = dihedral_point(c1, c2, c3, length, angle, dihed)
        t3 = time()
        coord_set[i0:] = c0
        t1 = time()
        global iit, dpt
        iit += t1-t0
        dpt += t3-t2

def cartesian_residue_interpolator(r, cartesian_atoms, dihedral_atoms):
        "Create a plan for Cartesian interpolation for all atoms."
        cartesian_atoms.extend(r.atoms)

def internal_residue_interpolator(r, cartesian_atoms, dihedral_atoms):
        """Create a plan for dihedral interpolation when possible,
        and Cartesian interpolation otherwise."""
        # First find the atoms that are connected to preceding
        # or succeeding residues.  If none, pick an arbitrary atom.
        # These atoms are always interpolated in Cartesian space.
        done = set()
        todo = []
        ratoms = r.atoms
        raset = set(ratoms)
        fixed = set()
        neighbor_res = set()
        neighbor_atoms = {a:a.neighbors for a in ratoms}
        c = r.chain
        if c:
                rba = (c.residue_before(r), c.residue_after(r))
                neighbor_res.update(rn for rn in rba if rn)
        if neighbor_res:
                for a0,natoms in neighbor_atoms.items():
                        for na in natoms:
                                if na.residue in neighbor_res:
                                        fixed.add(a0)
                                        break
        if not fixed:
                fixed.add(ratoms[0])
        for a0 in fixed:
                cartesian_atoms.append(a0)
                _finished(a0, done, todo, neighbor_atoms)

        # Now we look for atoms that are connected to those in
        # "fixed".  If we can find three atoms that define a
        # dihedral, we use dihedral interpolation; otherwise
        # we use Cartesian interpolation.
        while todo:
                na, a = todo.pop(0)
                if na in done:
                        # May be part of a loop and have been
                        # visited via another path
                        continue
                if na not in raset:
                        continue	# Atom from neighbor residue
                anchors = _findAnchor(a, done, neighbor_atoms)
                if len(anchors) >= 2:
                        # Found two anchor atoms connected to the
                        # fixed atom, we can use them for defining
                        # the dihedral
                        dihedral_atoms.extend((na, a, anchors[0], anchors[1]))
                        _finished(na, done, todo, neighbor_atoms)
                        continue
                if len(anchors) == 1:
                        # Found one anchor atom connected to the
                        # fixed atom, so we need to get another
                        # anchor atom connected to the one we found
                        # (but is not our original fixed atom)
                        anchors2 = _findAnchor(anchors[0], done, neighbor_atoms, a)
                        if len(anchors2) >= 1:
                                dihedral_atoms.extend((na, a, anchors[0], anchors2[0]))
                                _finished(na, done, todo, neighbor_atoms)
                                continue
                # Cannot find three fixed atoms to define dihedral.
                # Use Cartesian interpolation for this atom.
                cartesian_atoms.append(na)
                _finished(na, done, todo, neighbor_atoms)

        # Any left over atoms (usually disconnected) uses Cartesian interp
        for a0 in ratoms:
                if a0 not in done:
                        cartesian_atoms.append(a0)

def _finished(a, done, todo, neighbor_atoms):
        done.add(a)
        for na in neighbor_atoms[a]:
                if na not in done:
                        todo.append((na, a))

def _findAnchor(a, done, neighbor_atoms, ignore=None):
        anchors = []
        for na in neighbor_atoms[a]:
                if na in done and na is not ignore:
                        anchors.append(na)
        return anchors
