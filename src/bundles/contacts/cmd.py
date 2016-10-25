# vim: set expandtab ts=4 sw=4:

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

def contacts(session, atoms = None, probe_radius = 1.4, area_cutoff = 300):
    '''
    Compute buried solvent accessible surface areas between chains
    and show a 2-dimensional network graph depicting the contacts.

    Parameters
    ----------
    atoms : Atoms
    probe_radius : float
    '''
    sg = chain_spheres(atoms, session)			# List of SphereGroup
    ba = buried_areas(sg, probe_radius, area_cutoff)	# List of Contact

    # Remove common prefix for chain names.
    for g,sname in zip(sg,short_chain_names([g.full_name for g in sg])):
        g.name = sname

    # Report result
    areas = ['%s %s %.0f' % (c.group1.name, c.group2.name, c.buried_area) for c in ba]
    msg = '%d buried areas: ' % len(ba) + ', '.join(areas)
    log = session.logger
    log.info(msg)
    log.status(msg)

    if session.ui.is_gui:
        from . import tool
        tool.ContactPlot(session, sg, ba)
    else:
        log.warning("unable to show graph without GUI")

        
def register_contacts():
    from chimerax.core.commands import register, CmdDesc, AtomsArg, FloatArg
    desc = CmdDesc(
        optional = [('atoms', AtomsArg),],
        keyword = [('probe_radius', FloatArg),
                   ('area_cutoff', FloatArg),])
    register('contacts', desc, contacts)

from .graph import Node
class SphereGroup(Node):
    def __init__(self, name, atoms):
        self.full_name = self.name = name
        self.atoms = atoms
        self.centers = atoms.scene_coords
        self.radii = atoms.radii
        from numpy import mean
        self._color = mean(atoms.colors,axis=0)/255.0
        self._undisplayed_color = (.8,.8,.8,1)	# Node color for undisplayed chains
        self.area = None

    @property
    def size(self):
        return 0.03 * self.area		# Area of plot circle in pixels

    @property
    def position(self):
        return self.centroid()

    def shown(self):
        a = self.atoms
        return a.displays.any() or a.residues.ribbon_displays.any()

    @property
    def color(self):
        return self._color if self.shown() else self._undisplayed_color

    def centroid(self):
        return self.atoms.scene_coords.mean(axis = 0)

    def move(self, step):
        a = self.atoms
        if hasattr(self, '_original_coords'):
            a.coords = self._original_coords
        else:
            self._original_coords = a.coords
        a.scene_coords += step

    def unmove(self):
        if hasattr(self, '_original_coords'):
            self.atoms.coords = self._original_coords
            delattr(self, '_original_coords')

    def color_atoms(self, atoms, color):
        '''Restore original colors before coloring a subset of atoms.'''
        if hasattr(self, '_original_atom_colors'):
            self.atoms.colors = self._original_atom_colors
        else:
            self._original_atom_colors = self.atoms.colors
        atoms.colors = color

    def restore_atom_colors(self):
        if hasattr(self, '_original_atom_colors'):
            self.atoms.colors = self._original_atom_colors

def chain_spheres(atoms, session):
    if atoms is None:
        from chimerax.core.atomic import all_atoms
        atoms = all_atoms(session)
    if len(atoms) == 0:
        from chimerax.core.errors import UserError
        raise UserError('No atoms specified')
    from numpy import mean
    s = [SphereGroup('#%s/%s'%(m.id_string(),cid), catoms)
         for m, cid, catoms in atoms.by_chain]
    return s

def short_chain_names(names):
    use_short_names = (len(set(n.split('/',1)[0] for n in names)) == 1)
    sn = tuple(n.split('/',1)[-1] for n in names) if use_short_names else names
    return sn

def buried_areas(sphere_groups, probe_radius, min_area = 1):
    # Multi-threaded calculation of all pairwise buried areas.
    s = [(g, g.radii + probe_radius) for g in sphere_groups]
    s.sort(key = lambda v: len(v[1]), reverse = True)   # Biggest first for threading.
    
    # Compute area of each atom set.
    from chimerax.core.surface import spheres_surface_area
    from chimerax.core.threadq import apply_to_list
    def area(g, r):
        g.area = spheres_surface_area(g.centers,r).sum()
    apply_to_list(area, s)

    # Optimize buried area calculations using bounds of each atom set.
    naxes = 64
    from chimerax.core.geometry.sphere import sphere_points
    axes = sphere_points(naxes)
    from chimerax.core.geometry import sphere_axes_bounds
    bounds = [sphere_axes_bounds(g.centers, r, axes) for g, r in s]

    # Compute buried areas between all pairs.
    buried = []
    n = len(s)
    pairs = []
    from chimerax.core.geometry import bounds_overlap
    for i in range(n):
        for j in range(i+1,n):
            if bounds_overlap(bounds[i], bounds[j], 0):
                pairs.append((i,j))

    # Do multi-threaded buried area calculation.
    def barea(i, j, s = s, bounds = bounds, axes = axes, probe_radius = probe_radius):
        g1, r1 = s[i]
        g2, r2 = s[j]
        c = optimized_buried_area(g1.centers, r1, bounds[i], g2.centers, r2, bounds[j],
                                   axes, probe_radius)
        if c:
            c.group1, c.group2 = g1, g2
        return c
    bareas = apply_to_list(barea, pairs)

    # Apply minimum area threshold.
    buried = [c for c in bareas if c and c.buried_area >= min_area]

    # Sort to get predictable order since multithreaded calculation gives unpredictable ordering.
    buried.sort(key = lambda c: c.buried_area, reverse = True)

    return buried

# Consider only spheres in each set overlapping bounds of other set.
def optimized_buried_area(xyz1, r1, b1, xyz2, r2, b2, axes, probe_radius):

    # Check for no contact using bounding planes.
    # And find subsets of spheres that may be in contact to speed up area calculation.
    from chimerax.core.geometry import spheres_in_bounds
    i1 = spheres_in_bounds(xyz1, r1, axes, b2, 0)
    i2 = spheres_in_bounds(xyz2, r2, axes, b1, 0)
    if len(i1) == 0 or len(i2) == 0:
        return None

    # Compute areas for spheres near contact interface.
    xyz1, r1 = xyz1[i1], r1[i1]
    from chimerax.core.surface import spheres_surface_area
    a1 = spheres_surface_area(xyz1, r1)
    xyz2, r2 = xyz2[i2], r2[i2]
    a2 = spheres_surface_area(xyz2, r2)

    # Compute exposed areas for combined spheres.
    from numpy import concatenate
    xyz12, r12 = concatenate((xyz1,xyz2)), concatenate((r1,r2))
    a12 = spheres_surface_area(xyz12, r12)

    ba = 0.5 * (a1.sum() + a2.sum() - a12.sum())
    c = Contact(ba, a1, a2, a12, i1, i2)
    return c

from .graph import Edge
class Contact(Edge):
    def __init__(self, buried_area, area1, area2, area12, i1, i2):
        self.buried_area = buried_area
        self.area1i = area1	# Areas for atom index set i1
        self.area2i = area2	# Areas for atom index set i2
        self.area12i = area12
        self.i1 = i1
        self.i2 = i2
        self.group1 = None
        self.group2 = None

    @property
    def nodes(self):
        return (self.group1, self.group2)

    def contact_atoms(self, group, min_area = 1):
        g1, g2 = self.group1, self.group2
        n1 = len(self.area1i)
        if group is g1:
            ba = self.area1i - self.area12i[:n1]
            i = self.i1 if min_area is None else self.i1[ba >= min_area]
            atoms = g1.atoms[i]
        elif group is g2:
            ba = self.area2i - self.area12i[n1:]
            i = self.i2 if min_area is None else self.i2[ba >= min_area]
            atoms = g2.atoms[i]
        return atoms

    def contact_residue_atoms(self, group, min_area = 1):
        atoms = self.contact_atoms(group, min_area)
        return atoms.unique_residues.atoms

    def contact_residues(self, group, min_area = 15):
        g1, g2 = self.group1, self.group2
        n1 = len(self.area1i)
        if group is g1:
            i, areas = self.i1, self.area1i - self.area12i[:n1]
        elif group is g2:
            i, areas = self.i2, self.area2i - self.area12i[n1:]
        res, rareas = group.atoms[i].residue_sums(areas)
        return res.filter(rareas >= min_area)

    def explode_contact(self, distance = 30, move_group = None):
        g1, g2 = (self.group1, self.group2)
        xyz1, xyz2 = [self.contact_residue_atoms(g).scene_coords.mean(axis = 0) for g in (g1,g2)]
        from chimerax.core.geometry import normalize_vector
        step = (0.5*distance)*normalize_vector(xyz2 - xyz1)
        if move_group is g1:
            g1.move(-2*step)
            g2.unmove()
        elif move_group is g2:
            g1.unmove()
            g2.move(2*step)
        else:
            g1.move(-step)
            g2.move(step)

    def interface_frame(self, facing_group):
        r1, r2 = [self.contact_residues(g) for g in (self.group1, self.group2)]
        xyz1, xyz2 = [r.atoms.scene_coords.mean(axis = 0) for r in (r1,r2)]
        zaxis = (xyz2 - xyz1) if facing_group is self.group1 else (xyz1 - xyz2)
        center = 0.5 * (xyz1 + xyz2)
        from chimerax.core.geometry import orthonormal_frame
        f = orthonormal_frame(zaxis, origin = center)
        return f
        
def buried_area(xyz1, r1, a1, xyz2, r2, a2):

    from numpy import concatenate
    xyz12, r12 = concatenate((xyz1,xyz2)), concatenate((r1,r2))
    from chimerax.core.surface import spheres_surface_area
    a12 = spheres_surface_area(xyz12, r12).sum()
    ba = 0.5 * (a1 + a2 - a12)
    return ba

def neighbors(g, contacts):
    n = {}
    for c in contacts:
        if c.buried_area > 0:
            if c.group1 is g:
                n[c.group2] = c
            elif c.group2 is g:
                n[c.group1] = c
    return n
