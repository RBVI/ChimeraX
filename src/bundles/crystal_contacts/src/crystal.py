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

from math import pi

# -----------------------------------------------------------------------------
# Create a ball and stick model where balls represent asymmetric units of
# a crystal structure and sticks connect clashing asymmetric units.  Or make
# copies of contacting asymmetric units.
#
def show_crystal_contacts(molecule, dist,
                          make_copies = True,
                          rainbow = True,
                          schematic = False,
                          residue_info = False,
                          buried_areas = False,
                          probe_radius = 1.4,
                          intra_biounit = True,
                          angle_tolerance = pi/1800,   # 0.1 degrees
                          shift_tolerance = 0.1       # Angstroms
                          ):

    from chimerax.pdb_matrices import unit_cell_parameters
    if unit_cell_parameters(molecule) is None:
        from chimerax.core.errors import UserError
        raise UserError('No unit cell parameters for %s' % molecule.name)

    # Find all unique pairs of contacting asymmetric units.
    clist = report_crystal_contacts(molecule, dist,
                                    residue_info, buried_areas, probe_radius,
                                    intra_biounit,
                                    angle_tolerance, shift_tolerance)

    crystal = Crystal(molecule)

    # Incorrect matrices in PDB header may not be orthogonal.  Check.
    check_orthogonality(crystal.ncs_matrices, 'NCS', tolerance = 1e-3)
    check_orthogonality(crystal.smtry_matrices, 'SMTRY', tolerance = 1e-3)

    # Get list of clashing asymmetric units.
    asu_list = contacting_asymmetric_units(clist, crystal,
                                           include_000_unit_cell = True)

    cmodels = []
    
    # Place markers representing each asym unit.
    if schematic:
        marker_set = contact_marker_model(molecule, crystal, clist, asu_list)
        cmodels.append(marker_set)

    if make_copies:
        # Remove previous copies
        gm = copies_group_model(molecule, create=False)
        if gm:
            molecule.session.models.close([gm])

        # Make molecule copies.
        cm = make_asu_copies(molecule, contacting_asu(clist), crystal)
        if cm:
            molecule.session.models.add(cm, parent = copies_group_model(molecule))
            if rainbow:
                _rainbow_color_copies(cm)
            cmodels.extend(cm)

    # Zoom to show all asym units.
    molecule.session.main_view.view_all()

    return cmodels

# -----------------------------------------------------------------------------
# Non-crystallographic symmetry (NCS) asymmetric unit.
#
class NCS_Asymmetric_Unit:

    def __init__(self, ncs_index, smtry_index, unit_cell_index, transform):

        self.ncs_index = ncs_index
        self.smtry_index = smtry_index
        self.unit_cell_index = unit_cell_index
        self.transform = transform

# -----------------------------------------------------------------------------
# Symmetry matrices and crystal parameters from PDB header.
#
class Crystal:

    def __init__(self, molecule):

        from chimerax import pdb_matrices as m
        self.ncs_matrices = m.noncrystal_symmetries(molecule)
        self.smtry_matrices = m.crystal_symmetries(molecule)
        
        a, b, c, alpha, beta, gamma = m.unit_cell_parameters(molecule)[:6]
        from chimerax.crystal import unit_cell_axes
        self.axes = unit_cell_axes(a, b, c, alpha, beta, gamma)

        self.identity_smtry_index = identity_matrix_index(self.smtry_matrices)

    # -------------------------------------------------------------------------
    # Return transformation matrix for a specific asymmetric unit.
    #
    def transform(self, ncs_index, smtry_index, unit_cell_index):

        ncs = self.ncs_matrices[ncs_index]
        sym = self.smtry_matrices[smtry_index]
        sym_ncs = sym * ncs
        trans = unit_cell_translation(unit_cell_index, self.axes)
        trans_sym_ncs = translate_matrix(trans, sym_ncs)
        return trans_sym_ncs


# -----------------------------------------------------------------------------
# From list of clashing pairs of asymmetric units determine list of
# asymmetric units to display.
#
def contacting_asymmetric_units(clist, crystal, include_000_unit_cell = False):

    t = {}
    for atoms1, atoms2, equiv_asu_pairs in clist:
        for a1, a2 in equiv_asu_pairs:
            t[(a2.smtry_index,a2.unit_cell_index)] = 1
    if include_000_unit_cell:
        for si in range(len(crystal.smtry_matrices)):
            t[(si,(0,0,0))] = 1
    
    ncs_count = len(crystal.ncs_matrices)
    asu_list = []
    for si, uci in t.keys():
        for ni in range(ncs_count):
            tf = crystal.transform(ni, si, uci)
            asu_list.append(NCS_Asymmetric_Unit(ni, si, uci, tf))
    
    return asu_list

# -----------------------------------------------------------------------------
#
def contact_marker_model(molecule, crystal, clist, asu_list):

    # Set marker and link colors and radii for clash depiction.
    xyz = molecule.atoms.coords
    asu_center = xyz.mean(axis = 0)
    bbox = bounding_box(xyz)
    radius_ncs = .25*min([a-b for a, b in zip(bbox[1], bbox[0])])
    radius_copies = .8 * radius_ncs
    rgba_ncs = (128, 255, 128, 255)   # NCS asym unit, light green
    rgba_uc = (128, 205, 230, 255)   # Unit cell, light blue
    rgba_ouc = (230, 230, 128, 255)  # Other unit cells, light yellow
    rgba_overlap = rgba_ncs[:3] + (128,)
    link_rgba = (255, 128, 128, 255)
    link_radius = radius_copies
    uc_thickness = .4 * radius_ncs

    # Create markers and links depicting asym units and clashes.
    from chimerax.markers import MarkerSet
    marker_set = MarkerSet(molecule.session, molecule.name + ' crystal contacts schematic')
    markers = create_asymmetric_unit_markers(marker_set, asu_list, asu_center,
                                             crystal.identity_smtry_index,
                                             rgba_ncs, rgba_uc, rgba_ouc,
                                             radius_ncs, radius_copies)
    n = len(molecule.atoms)
    links = create_clash_links(clist, n, link_radius, link_rgba, markers)

    # Make marker transparent when clashing asym unit are close.
    for l in links:
        if l.length < .5 * radius_ncs:
            l.atoms[0].color = rgba_overlap

    create_unit_cell(marker_set, crystal.axes, rgba_uc, uc_thickness)

    molecule.session.models.add([marker_set])
    
    return marker_set
    
# -----------------------------------------------------------------------------
# Create a sphere representing each asymmetric unit.  The sphere is created
# using the volume path tracer module.
#
def create_asymmetric_unit_markers(marker_set, asu_list, asu_center,
                                   identity_smtry_index,
                                   rgba_ncs, rgba_uc, rgba_ouc,
                                   radius_ncs, radius_copies):

    markers = {}
    for asu in asu_list:
        c = asu.transform * asu_center
        if asu.unit_cell_index != (0,0,0):
            color = rgba_ouc
            r = radius_copies
        elif asu.smtry_index == identity_smtry_index:
            color = rgba_ncs
            r = radius_ncs
        else:
            color = rgba_uc
            r = radius_copies
        m = marker_set.create_marker(c, color, r)
        m.name = ('n%d s%d %d %d %d' %
                       ((asu.ncs_index, asu.smtry_index) + asu.unit_cell_index))
        markers[(asu.ncs_index, asu.smtry_index, asu.unit_cell_index)] = m

    return markers

# -----------------------------------------------------------------------------
# Create cylinders connecting spheres representing asymmetric unit that
# have close atomic contacts.  The cylinders are volume path tracer links.
#
def create_clash_links(clist, n, link_radius, link_rgba, markers):

    from chimerax.markers import create_link
    links = []
    for atoms1, atoms2, equiv_asu_pairs in clist:
        rscale = 20*float(len(atoms1))/n
        if rscale > 1: rscale = 1
        elif rscale < .2: rscale = .2
        for a1,a2 in equiv_asu_pairs:
            m1 = markers[(a1.ncs_index, a1.smtry_index, a1.unit_cell_index)]
            m2 = markers[(a2.ncs_index, a2.smtry_index, a2.unit_cell_index)]
            l = create_link(m1, m2, link_rgba, link_radius * rscale)
            links.append(l)
    return links

# -----------------------------------------------------------------------------
# Create markers and links outlining unit cell.
#
def create_unit_cell(marker_set, axes, color, thickness):

    r = 0.5 * thickness

    corner_indices = ((0,0,0), (0,0,1), (0,1,1), (0,1,0),
                      (1,0,0), (1,0,1), (1,1,1), (1,1,0))
    corners = [unit_cell_translation(i, axes) for i in corner_indices]
    
    markers = []
    for c in corners:
        m = marker_set.create_marker(c, color, r)
        markers.append(m)

    edge_indices = ((0,1),(1,2),(2,3),(3,0),
                    (4,5),(5,6),(6,7),(7,4),
                    (0,4),(1,5),(2,6),(3,7))
    from chimerax.markers import create_link
    for i,j in edge_indices:
        create_link(markers[i], markers[j], color, r)

# -----------------------------------------------------------------------------
# Determine clashing asymmetric units and print a text list of clashing pairs
# having unique relative orientations.
#
def report_crystal_contacts(molecule, distance,
                            residue_info = False,
                            buried_areas = False,
                            probe_radius = 1.4,
                            intra_biounit = True,
                            angle_tolerance = pi/1800,   # 0.1 degrees
                            shift_tolerance = 0.1):      # Angstroms

    clist = asymmetric_unit_contacts(molecule, distance, intra_biounit,
                                     angle_tolerance, shift_tolerance)
    lines = []
    lines.append('%d pairs of asymmetric units of %s contact at distance %.1f A' %
                 (len(clist), molecule.name, distance))
    if clist:
        clist.sort(key = lambda c: len(c[0]))
        clist.reverse()
        lines.append(asu_contacts_info(clist))

        if residue_info:
            rc = residue_contacts(clist, distance)
            lines.append(residue_contacts_info(rc))
        if buried_areas:
            if not residue_info:
                rc = residue_contacts(clist, distance)
            lines.append(buried_areas_info(rc, probe_radius))

    msg = '\n'.join(lines)
    molecule.session.logger.info(msg, is_html = True)
    
    return clist

# -----------------------------------------------------------------------------
#
def asu_contacts_info(clist):

    lines = ['<pre>',
             '  Atoms  MTRIX  SMTRY   Unit cell  MTRIXref  Copies']
    for atoms1, atoms2, equiv_asu_pairs in clist:
        a1, a2 = equiv_asu_pairs[0]
        na = len(atoms1)
        n1 = a1.ncs_index
        n2, s2, uc2 = a2.ncs_index, a2.smtry_index, a2.unit_cell_index
        ne = len(equiv_asu_pairs)
        lines.append('%6d   %3d    %3d     %2d %2d %2d    %3d    %6d' %
                     (na, n2, s2, uc2[0], uc2[1], uc2[2], n1, ne))
    lines.append('</pre>')
    return '\n'.join(lines)

# -----------------------------------------------------------------------------
#
def residue_contacts_info(rcontacts):

    lines = ['<pre>',
             'Residue contacts, %d residues' % len(rcontacts),
             '  residue1  ncs1  residue2  ncs2  sym2    cell2  distance']
    for r1, asu1, r2list in rcontacts:
        r1name = '%s %d %s' % (r1.name, r1.number, r1.chain_id)
        for r2,asu2,d in r2list:
            r2name = '%s %d %s' % (r2.name, r2.number, r2.chain_id)
            uc2 = asu2.unit_cell_index
            lines.append('%10s  %3d %10s  %3d   %3d   %2d %2d %2d   %.3g'
                         % (r1name, asu1.ncs_index, r2name,
                            asu2.ncs_index, asu2.smtry_index, uc2[0], uc2[1], uc2[2], d))
    lines.append('</pre>')
    return '\n'.join(lines)

# -----------------------------------------------------------------------------
#
def buried_areas_info(rcontacts, probe_radius):

    areas = []
    for r1, asu1, r2list in rcontacts:
        a = buried_area(r1, asu1.transform,
                        [(r2, asu2.transform) for r2, asu2, d in r2list],
                        probe_radius)
        areas.append((r1, asu1, a))

    lines =['<pre>',
            'Residue buried areas, %d residues' % len(areas),
            '  residue    ncs   buried']
    for r1, asu1, a in areas:
        r1name = '%s %d %s' % (r1.name, r1.number, r1.chain_id)
        lines.append('%10s  %3d     %.3g' % (r1name, asu1.ncs_index, a))
        r1.maxCrystalBuriedArea = max(a, getattr(r1,'maxCrystalBuriedArea',0))
    lines.append('</pre>')
    return '\n'.join(lines)

# -----------------------------------------------------------------------------
#
def buried_area(r, tf, rlist, probe_radius = 1.4):

    xyz12, r12 = residue_xyzr([(r,tf)] + rlist)
    na = r.num_atoms
    xyz1, r1 = xyz12[:na], r12[:na]

    from chimerax.surface import spheres_surface_area
    a1a = spheres_surface_area(xyz1, r1)
    a12a = spheres_surface_area(xyz12, r12)
    bsas = (a1a[:na] - a12a[:na]).sum()

    return bsas

# -----------------------------------------------------------------------------
#
def residue_xyzr(rtflist):

    n = sum([r.num_atoms for r,tf in rtflist])
    from numpy import empty, float32
    xyz = empty((n,3), float32)
    r = empty((n,), float32)
    b = 0
    for res,tf in rtflist:
        ratoms = res.atoms
        n = len(ratoms)
        xyz[b:b+n,:3] = ratoms.coords
        r[b:b+n] = ratoms.radii
        tf.transform_points(xyz[b:b+n], in_place = True)
        b += n
    return xyz, r

# -----------------------------------------------------------------------------
#
def residue_contacts(clist, distance):

    rclose = {}
    for atoms1, atoms2, equiv_asu_pairs in clist:
        asu1, asu2 = equiv_asu_pairs[0]
        apairs = close_atom_pairs(atoms1, asu1, atoms2, asu2, distance)
        rpairs = close_residue_pairs(apairs)
        for r1,r2,d in rpairs:
            if (r1,asu1) in rclose:
                rclose[(r1,asu1)].append((r2,asu2,d))
            else:
                rclose[(r1,asu1)] = [(r2,asu2,d)]
    rc = [(r1, asu1, r2list) for (r1,asu1), r2list in rclose.items()]
    rc.sort(key = lambda o: (o[0].chain_id, o[0].number))
    return rc

# -----------------------------------------------------------------------------
#
def close_atom_pairs(atoms1, asu1, atoms2, asu2, distance):

    xyz1 = atoms1.coords
    asu1.transform.transform_points(xyz1, in_place=True)
    xyz2 = atoms2.coords
    asu2.transform.transform_points(xyz2, in_place=True)

    close = []
    import  numpy
    dist = numpy.zeros((len(atoms2),), numpy.float32)
    from chimerax.geometry import distances_from_origin
    for i1, a1 in enumerate(atoms1):
        distances_from_origin(xyz2, xyz1[i1,:], dist)
        for i2, a2 in enumerate(atoms2):
            if dist[i2] <= distance:
                close.append((a1, a2, dist[i2]))

    return close

# -----------------------------------------------------------------------------
#
def close_residue_pairs(apairs):

    rdist = {}
    for a1, a2, dist in apairs:
        rp = (a1.residue, a2.residue)
        if rp in rdist:
            rdist[rp] = min(rdist[rp], dist)
        else:
            rdist[rp] = dist
    rpairs = [(rp[0], rp[1], d) for rp, d in rdist.items()]
    return rpairs

# -----------------------------------------------------------------------------
# Angle and shift tolerance values are used to eliminate equivalent pairs
# of asymmetric units. To find only contacts between different biological
# units (defined by PDB BIOMT matrices) use intra_biounit = False.
#
def asymmetric_unit_contacts(molecule, distance, intra_biounit,
                             angle_tolerance, shift_tolerance):

    plist = nearby_asymmetric_units(molecule, distance)
    if not intra_biounit:
        plist = interbiounit_asu_pairs(molecule, plist,
                                       angle_tolerance, shift_tolerance)
    uplist = unique_asymmetric_unit_pairs(plist, angle_tolerance,
                                          shift_tolerance)
    
    alist = molecule.atoms
    from numpy import float32
    xyz = alist.coords.astype(float32)

    catoms = []
    for equiv_asu_pairs in uplist:
        asu1, asu2 = equiv_asu_pairs[0]
        t1 = asu1.transform.matrix
        t2 = asu2.transform.matrix
        from chimerax.geometry import find_close_points_sets
        i1, i2 = find_close_points_sets([(xyz, t1)], [(xyz, t2)], distance)
        if len(i1[0]) > 0:
            atoms1 = alist.filter(i1[0])
            atoms2 = alist.filter(i2[0])
            catoms.append((atoms1, atoms2, equiv_asu_pairs))

    return catoms

# -----------------------------------------------------------------------------
# Find nearby asymmetric units in a crystal by looking for overlapping bounding
# boxes.  Contacts with each non-crystallographic symmetry (NCS) position
# are considered.  Returns a list of triples, the first item being the NCS
# matrix, the second being a another matrix for symmetry placing an asymmetric
# unit nearby, and the third being identifiers for these two matrices
# positions (NCS1, NCS2, SMTRY2, unitcell2).  The matrices are relative to
# the given molecule's local coordinates.
#
def nearby_asymmetric_units(molecule, distance):

    from chimerax.pdb_matrices import unit_cell_parameters
    if unit_cell_parameters(molecule) is None:
        return []

    xyz = molecule.atoms.coords
    bbox = bounding_box(xyz)
    pbox = pad_box(bbox, .5 *distance)

    crystal = Crystal(molecule)

    plist = nearby_boxes(pbox, crystal)
    
    return plist

# -----------------------------------------------------------------------------
# Filter out pairs of asymmetric units that are within the same biological unit
# defined by PDB BIOMT matrices.
#
def interbiounit_asu_pairs(molecule, plist, angle_tolerance, shift_tolerance):

    # Get biomt matrices.
    from chimerax.pdb_matrices import biological_unit_matrices
    biomt = biological_unit_matrices(molecule)
    if len(biomt) <= 1:
        return plist

    # Make sure biomt matrices contain identity.
    # This finds all relative biomt transforms.  We assume they form a group
    # (in mathematical sense, closed under multiplication) otherwise it isn't
    # clear how the biounits are layed out in the crystal.
    # TODO: Check group assumption and warn if does not hold.
    have_identity = False
    for tf in biomt:
        if tf.is_identity():
            have_identity = True
            break
    if not have_identity:
        tfinv = biomt[0].inverse()
        biomt = [tfinv * tf for tf in biomt]

    # Bin relative biomt matrices for fast approximate equality test.
    from chimerax.geometry.bins import Binned_Transforms
    b = Binned_Transforms(angle_tolerance, shift_tolerance)
    for tf in biomt:
        b.add_transform(tf)

    # Filter out asu pairs that belong to same biological unit.
    ilist = []
    for asu1, asu2 in plist:
        rel = asu1.transform.inverse() * asu2.transform
        if len(b.close_transforms(rel)) == 0:
            ilist.append((asu1, asu2))

    return ilist

# -----------------------------------------------------------------------------
# Group pairs of asymmetric units that have the same relative orientation
# together.
#
def unique_asymmetric_unit_pairs(plist, angle_tolerance, shift_tolerance):

    from chimerax.geometry.bins import Binned_Transforms
    b = Binned_Transforms(angle_tolerance, shift_tolerance)

    uplist = []
    tf1_inverse_cache = {}
    equiv_asu_pairs = {}
    for asu1, asu2 in plist:
        tf1_index = asu1.ncs_index
        if tf1_index in tf1_inverse_cache:
            tf1_inv = tf1_inverse_cache[tf1_index]
        else:
            tf1_inv = asu1.transform.inverse()
            tf1_inverse_cache[tf1_index] = tf1_inv
        rel = tf1_inv * asu2.transform
        close = b.close_transforms(rel)
        if close:
            equiv_asu_pairs[id(close[0])].append((asu1, asu2))    # duplicate
        else:
            b.add_transform(rel)
            apair = [(asu1, asu2)]
            equiv_asu_pairs[id(rel)] = apair
            uplist.append(apair)

    return uplist

# -----------------------------------------------------------------------------
# Apply crystal symmetry to given box and find all boxes that intersect any
# NCS symmetry of the given box.
#
def nearby_boxes(box, crystal):

    act = basis_coordinate_transform(crystal.axes)
    isi = crystal.identity_smtry_index

    box2_cache = {}
    
    plist = []
    for ncs1index, ncs1 in enumerate(crystal.ncs_matrices):
        ncs1_inv = ncs1.inverse()
        box1 = transformed_bounding_box(box, act * ncs1)
        asu1 = NCS_Asymmetric_Unit(ncs1index, isi, (0,0,0), ncs1)
        for ncs2index, ncs2 in enumerate(crystal.ncs_matrices):
            for symindex, sym in enumerate(crystal.smtry_matrices):
                identity_sym = (symindex == isi)
                if (ncs2index, symindex) in box2_cache:
                    sym_ncs2, box2 = box2_cache[(ncs2index, symindex)]
                else:
                    sym_ncs2 = sym * ncs2
                    box2_tf = act * sym_ncs2
                    box2 = transformed_bounding_box(box, box2_tf)
                    box2_cache[(ncs2index, symindex)] = (sym_ncs2, box2)
                tlist = overlapping_translations(box1, box2, crystal.axes)
                for t, ucijk in tlist:
                    if (identity_sym and ucijk == (0,0,0) and
                        ncs1index >= ncs2index):
                        continue        # Include only 1 copy of pair
                    trans_sym_ncs2 = translate_matrix(t, sym_ncs2)
                    asu2 = NCS_Asymmetric_Unit(ncs2index, symindex, ucijk,
                                               trans_sym_ncs2)
                    plist.append((asu1, asu2))

    return plist

# -----------------------------------------------------------------------------
# Boxes are in crystal axes coordinates.
#
def overlapping_translations(box1, box2, axes):

    from math import ceil, floor
    tintervals = []
    for a in range(3):
        t0 = int(ceil(box1[0][a]-box2[1][a]))
        t1 = int(floor(box1[1][a]-box2[0][a]))
        if t0 > t1:
            return []
        tintervals.append(range(t0, t1+1))

    ar, br, cr = tintervals
    tlist = []
    for i in ar:
        for j in br:
            for k in cr:
                t = unit_cell_translation((i,j,k), axes)
                tlist.append((t,(i,j,k)))
    return tlist

# -----------------------------------------------------------------------------
# Transformation from xyz position to unit cell indices.
#
def basis_coordinate_transform(axes):

    from chimerax.geometry import Place
    bct = Place(((axes[0][0], axes[1][0], axes[2][0], 0),
                 (axes[0][1], axes[1][1], axes[2][1], 0),
                 (axes[0][2], axes[1][2], axes[2][2], 0))).inverse()
    return bct

# -----------------------------------------------------------------------------
# Translation vector for a unit cell with given indices.
#
def unit_cell_translation(ijk, axes):

    i,j,k = ijk
    t = (i*axes[0][0]+j*axes[1][0]+k*axes[2][0],
         i*axes[0][1]+j*axes[1][1]+k*axes[2][1],
         i*axes[0][2]+j*axes[1][2]+k*axes[2][2])
    return t

# -----------------------------------------------------------------------------
#
def bounding_box(xyz):

    from chimerax.geometry import point_bounds
    b = point_bounds(xyz)
    return b.xyz_min, b.xyz_max
    
# -----------------------------------------------------------------------------
#
def box_center(box):

    c = [.5*(a+b) for a,b in zip(box[0], box[1])]
    return c
    
# -----------------------------------------------------------------------------
#
def pad_box(box, padding):

    xyz_min = [x-padding for x in box[0]]
    xyz_max = [x+padding for x in box[1]]
    pbox = (xyz_min, xyz_max)
    return pbox
    
# -----------------------------------------------------------------------------
#
def transformed_bounding_box(box, tf):

    (x0,y0,z0), (x1,y1,z1) = box
    from numpy import array, float32
    corners = array(((x0,y0,z0), (x0,y0,z1), (x0,y1,z0), (x0,y1,z1),
                     (x1,y0,z0), (x1,y0,z1), (x1,y1,z0), (x1,y1,z1)), float32)
    tf_corners = tf.transform_points(corners)
    tf_box = bounding_box(tf_corners)
    return tf_box

# -----------------------------------------------------------------------------
#
def translate_matrix(t, m):

    from chimerax.geometry import translation
    return translation(t) * m
    
# -----------------------------------------------------------------------------
#
def identity_matrix_index(matrices):

    for i in range(len(matrices)):
        if matrices[i].is_identity():
            return i
    return None
    
# -----------------------------------------------------------------------------
#
def check_orthogonality(matrices, name, tolerance):

    for mindex, m in enumerate(matrices):
        mr = m.zero_translation()
        mrt = mr.transpose()
        p = mr * mrt
        if not p.is_identity(tolerance):
            print ('%s matrix %d is not orthogonal, tolerance %.3g' %
                   (name, mindex, tolerance))
            print_matrix(m, '%10.5f')
            print ('  matrix times transpose = ')
            print_matrix(p, '%10.5f')

# -----------------------------------------------------------------------------
#
def print_matrix(m, format):

    lformat = ' '.join(['%10.5f']*4)
    for r in m.matrix:
        print (lformat % tuple(r))

# -----------------------------------------------------------------------------
#
def contacting_asu(clist):

    asu = {}
    for atoms1, atoms2, equiv_asu_pairs in clist:
        for a in equiv_asu_pairs[0]:
            ai = (a.ncs_index, a.smtry_index, a.unit_cell_index)
            if ai != (0,0,(0,0,0)):
                asu[ai] = a
    return asu.values()

# -----------------------------------------------------------------------------
#
def make_asu_copies(m, asu_list, crystal):

    xflist = []
    names = []
    for asu in asu_list:
        if asu.transform.is_identity():
            continue
        xflist.append(asu.transform)
        name = '%s %s' % (m.name, '%d %d %d' % asu.unit_cell_index)
        if len(crystal.smtry_matrices) > 1:
            name += ' sym %d' % asu.smtry_index
        if len(crystal.ncs_matrices) > 1:
            name += ' ncs %d' % asu.ncs_index
        names.append(name)

    cmodels = make_molecule_copies(m, xflist, names)

    return cmodels

# -----------------------------------------------------------------------------
#
def _rainbow_color_copies(models):
    if len(models) == 0:
        return
    from chimerax.std_commands.rainbow import rainbow
    from chimerax.core.objects import Objects
    rainbow(models[0].session, Objects(models = models), 'structures')

# -----------------------------------------------------------------------------
#
def make_molecule_copies(m, xflist, names):

    mclist = []
    for c, xf in enumerate(xflist):
        mc = m.copy()
        mclist.append(mc)
        mc.name = names[c]
        mc.scene_position = m.scene_position * xf
        mc._crystal_contacts_copy = True
    return mclist

# -----------------------------------------------------------------------------
#
def copies_group_model(m, create = True):
    name = '%s crystal contacts' % m.name
    gmodels = [g for g in m.session.models.list() if g.name == name]
    if gmodels:
        gm = gmodels[0]
    elif create:
        from chimerax.core.models import Model
        gm = Model(name, m.session)
        gm.model_panel_show_expanded = False
        m.session.models.add([gm])
    else:
        gm = None
    return gm

# -----------------------------------------------------------------------------
#
def schematic_model(m):
    name = '%s crystal contacts schematic' % m.name
    gmodels = [g for g in m.session.models.list() if g.name == name]
    return gmodels[0] if gmodels else None

