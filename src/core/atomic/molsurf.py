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

"""
molsurf: Compute molecular surfaces
===================================
"""

from ..models import Model

class MolecularSurface(Model):
    '''
    A molecular surface computed from a set of atoms.
    This can be a solvent excluded surface which is the
    boundary of the region where a sphere of radius
    *probe_radius* cannot reach.  Or it can be a contour
    surface of a density map created by placing Gaussian
    shapes at the center of each atom position where
    the *resolution* parameter controls the width of the Gaussian.
    Part of a surface can be undisplayed, leaving only the
    patch associated with *show_atoms* :class:`.Atoms`.

    Parameters
    ----------
    session : :class:`~chimerax.core.session.Session`
      The session the surface model will belong to
    enclose_atoms : :class:`.Atoms`
      Surface bounds these atoms.
    show_atoms : :class:`.Atoms`
      Show only the portion of the surface near these atoms.
    probe_radius : float
      The probe sphere radius for a solvent excluded surface.
    grid_spacing : float
      Spacing of 3-dimensional grid used in surface calculations.
      Typical value os 0.5 Angstroms. Finer values give smoother surfaces
      but take more memory and longer computation times.
    resolution : float
      Used only for Gaussian surfaces, specifies a nominal density
      map resolution.  See the :func:`.molmap` for details.
    level : float or None
      Threshold level for Gaussian surfaces. The density map used to
      compute these surface uses Gaussian heights equal to atomic number
      so the level is in atomic number units.  If None is specified then
      the level chosen is the minimum density at the atom positions.
    name : string
      Surface name.

    color : numpy uint8 length 4 array
      RGBA color for surface.
    visible_patches : int or None
      Number of connected surface components to show.
      Only the largest area N components will be shown.
      For value None all components are shown.
    sharp_boundaries : bool
      Whether to subdivide triangles composing the surface so that
      triangle edges lie exactly between atoms. This creates less jagged
      edges when showing or coloring patches of surfaces for a subset of atoms.
    '''

    def __init__(self, session, enclose_atoms, show_atoms, probe_radius, grid_spacing,
                 resolution, level, name, color, visible_patches, sharp_boundaries):
        
        Model.__init__(self, name, session)

        self.atoms = enclose_atoms
        self.show_atoms = show_atoms	# Atoms for surface patch to show
        self.probe_radius = probe_radius # Only used for solvent excluded surface
        self.grid_spacing = grid_spacing
        self.resolution = resolution    # Only used for Gaussian surface
        self.level = level		# Contour level for Gaussian surface, atomic number units
        self.color = color
        self.visible_patches = visible_patches
        self.sharp_boundaries = sharp_boundaries
        self._refinement_steps = 1	# Used for fixing sharp edge problems near 3 atom junctions.

        self._vertex_to_atom = None
        self._vertex_to_atom_count = None	# Used to check if atoms deleted
        self._max_radius = None
        self.clip_cap = True

    def new_parameters(self, show_atoms, probe_radius = None, grid_spacing = None,
                       resolution = None, level = None, visible_patches = None, sharp_boundaries = None,
                       color = None, transparency = None):
        '''
        Change the surface parameters.  Parameter definitions are the
        same as for the contructor.
        '''

        shown_changed = show_atoms.hash() != self.show_atoms.hash()
        self.show_atoms = show_atoms	# Atoms for surface patch to show
        
        shape_change = False
        if probe_radius is not None and probe_radius != self.probe_radius:
            self.probe_radius = probe_radius
            shape_change = True
        if grid_spacing is not None and grid_spacing != self.grid_spacing:
            self.grid_spacing = grid_spacing
            shape_change = True
        if resolution is not None and resolution != self.resolution:
            self.resolution = None if resolution <= 0 else resolution
            shape_change = True
        if level is not None and level != self.level:
            self.level = level
            shape_change = True
        if visible_patches is not None and visible_patches != self.visible_patches:
            self.visible_patches = visible_patches
            shown_changed = True
        if sharp_boundaries is not None and sharp_boundaries != self.sharp_boundaries:
            self.sharp_boundaries = sharp_boundaries
            shape_change = True

        if shape_change:
            self.vertices = None
            self.normals = None
            self.triangles = None
            self.color = self._average_color()
            self.vertex_colors = None
            self._vertex_to_atom = None
            self._max_radius = None
        elif shown_changed:
            self.triangle_mask = self._calc_triangle_mask()
                
    @property
    def atom_count(self):
        '''Number of atoms for calculating the surface. Read only.'''
        return len(self.atoms)

    def atom_coords(self):
        atoms = self.atoms
        return atoms.coords if atoms.single_structure else atoms.scene_coords
    
    def calculate_surface_geometry(self):
        '''Recalculate the surface if parameters have been changed.'''
        if not self.vertices is None:
            return              # Geometry already computed

        atoms = self.atoms
        xyz = self.atom_coords()
        res = self.resolution
        from .. import surface
        if res is None:
            # Compute solvent excluded surface
            r = atoms.radii
            self._max_radius = r.max()
            va, na, ta = surface.ses_surface_geometry(xyz, r, self.probe_radius, self.grid_spacing)
        else:
            # Compute Gaussian surface
            va, na, ta, level = surface.gaussian_surface(xyz, atoms.element_numbers, res,
                                                         self.level, self.grid_spacing)
            self.gaussian_level = level

        if self.sharp_boundaries:
            v2a = self.vertex_to_atom_map(va)
            kw = {'refinement_steps': self._refinement_steps}
            if self.resolution is None:
                kw['atom_radii'] = atoms.radii
            from ..surface import sharp_edge_patches
            va, na, ta, tj, v2a = sharp_edge_patches(va, na, ta, v2a, xyz, **kw)
            self.joined_triangles = tj	# With non-duplicate vertices for clip cap calculation
            self._vertex_to_atom = v2a

        self.vertices = va
        self.normals = na
        self.triangles = ta
        self.triangle_mask = self._calc_triangle_mask()
        self.update_selection()

    def _calc_triangle_mask(self):
        tmask = self._patch_display_mask(self.show_atoms)
        if self.visible_patches is None:
            return tmask

        from .. import surface
        if self.sharp_boundaries:
            # With sharp boundaries triangles are not connected.
            vmap = surface.unique_vertex_map(self.vertices)
            tri = vmap[self.triangles]
        else:
            tri = self.triangles
        m = surface.largest_blobs_triangle_mask(self.vertices, tri, tmask,
                                                blob_count = self.visible_patches,
                                                rank_metric = 'area rank')
        return m

    def vertex_to_atom_map(self, vertices = None):
        '''
        Returns a numpy array of integer values with length equal to
        the number of surface vertices and value is the atom index for
        the atom closest to each vertex.  Can return None if atoms are
        not associated with vertices.  Supplying vertices argument computes
        new vertex to atom map.
        '''
        if vertices is not None:
            xyz1 = self.vertices if vertices is None else vertices
            xyz2 = self.atom_coords()
            radii = {'scale2':self.atoms.radii} if self.resolution is None else {}
            max_dist = self._maximum_atom_to_surface_distance()
            from .. import geometry
            i1, i2, nearest1 = geometry.find_closest_points(xyz1, xyz2, max_dist, **radii)
            if len(i1) < len(xyz1):
                # TODO: For Gaussian surface should increase max_dist and try again.
                raise RuntimeError('Surface further from atoms than expected (%g) for %d of %d atoms'
                                   % (max_dist, len(xyz1)-len(i1), len(xyz1)))
            from numpy import empty, int32
            v2a = empty((len(xyz1),), int32)
            v2a[i1] = nearest1
            self._vertex_to_atom = v2a
            self._vertex_to_atom_count = len(self.atoms)
        elif self._vertex_to_atom is not None and len(self.atoms) < self._vertex_to_atom_count:
            # Atoms deleted
            self._vertex_to_atom = None
        return self._vertex_to_atom

    def _maximum_atom_to_surface_distance(self):
        res = self.resolution
        if res is None:
            d = 1.1 * (self.probe_radius + self._max_radius + self.grid_spacing)
        else:
            d = 2*(res + self.grid_spacing)
        return d

    def _patch_display_mask(self, patch_atoms):
        surf_atoms = self.atoms
        if len(patch_atoms) == len(surf_atoms):
            return None
        shown_atoms = surf_atoms.mask(patch_atoms)
        return self._atom_triangle_mask(shown_atoms)

    def _atom_triangle_mask(self, atom_mask):
        v2a = self.vertex_to_atom_map()
        if v2a is None:
            return None
        shown_vertices = atom_mask[v2a]
        t = self.triangles
        from numpy import logical_and, empty, bool
        shown_triangles = empty((len(t),), bool)
        logical_and(shown_vertices[t[:,0]], shown_vertices[t[:,1]], shown_triangles)
        logical_and(shown_triangles, shown_vertices[t[:,2]], shown_triangles)
        return shown_triangles

    def has_atom_patches(self):
        return self.vertex_to_atom_map() is not None
        
    def show(self, atoms, only = False):
        '''
        Show the surface patch near these :class:`.Atoms` in
        addition to any already shown surface patch.
        '''
        if self.has_atom_patches():
            self.show_atoms = atoms if only else self.show_atoms.merge(atoms)
            self.triangle_mask = self._calc_triangle_mask()
            self.display = True
        elif len(atoms) == len(self.atoms):
            self.display = True
            self.triangle_mask = None

    def hide(self, atoms):
        '''
        Hide the surface patch near these :class:`.Atoms`.
        '''
        if self.has_atom_patches():
            self.show_atoms = self.show_atoms.subtract(atoms)
            self.triangle_mask = self._calc_triangle_mask()
        elif len(atoms) == len(self.atoms):
            self.display = False
            self.triangle_mask = None

    def _get_single_color(self):
        vc = self.vertex_colors
        from ..colors import most_common_color
        return self.color if vc is None else most_common_color(vc)
    def _set_single_color(self, color):
        self.color = color
        self.vertex_colors = None
    single_color = property(_get_single_color, _set_single_color)

    def _average_color(self):
        vc = self.vertex_colors
        if vc is None or len(vc) == 0:
            return self.color
        from numpy import float32, uint8
        csum = vc.sum(axis = 0, dtype = float32)
        csum /= len(vc)
        acolor = csum.astype(uint8)
        return acolor

    def first_intercept(self, mxyz1, mxyz2, exclude = None):
        # Pick atom associated with surface patch
        from ..graphics import Drawing
        p = Drawing.first_intercept(self, mxyz1, mxyz2, exclude)
        if p is None:
            return None
        t = p.triangle_number
        v = self.triangles[t,0]
        v2a = self.vertex_to_atom_map()
        if v2a is None:
            pa = p
        else:
            a = v2a[v]
            atom = self.atoms[a]
            from .structure import PickedAtom
            pa = PickedAtom(atom, p.distance)
        return pa

    def update_selection(self):
        asel = self.atoms.selected
        tmask = self._atom_triangle_mask(asel)
        if tmask is None:
            self.selected = False
        else:
            self.selected = (tmask.sum() > 0)
        self.selected_triangles_mask = tmask

    # State save/restore in ChimeraX
    _save_attrs = ('_refinement_steps', '_vertex_to_atom', '_vertex_to_atom_count', '_max_radius',
                   'vertices', 'normals', 'triangles', 'triangle_mask', 'vertex_colors', 'color',
                   'joined_triangles')

    def take_snapshot(self, session, flags):
        init_attrs = ('atoms', 'show_atoms', 'probe_radius', 'grid_spacing', 'resolution', 'level',
                      'name', 'color', 'visible_patches', 'sharp_boundaries')
        data = {attr:getattr(self, attr) for attr in init_attrs}
        data['model state'] = Model.take_snapshot(self, session, flags)
        data.update({attr:getattr(self,attr) for attr in self._save_attrs if hasattr(self,attr)})
        from ..state import CORE_STATE_VERSION
        data['version'] = CORE_STATE_VERSION
        return data

    @staticmethod
    def restore_snapshot(session, data):
        d = data
        s = MolecularSurface(session, d['atoms'], d['show_atoms'],
                             d['probe_radius'], d['grid_spacing'], d['resolution'],
                             d['level'], d['name'], d['color'], d['visible_patches'],
                             d['sharp_boundaries'])
        Model.set_state_from_snapshot(s, session, d['model state'])
        for attr in MolecularSurface._save_attrs:
            setattr(s, attr, d[attr])

    def reset_state(self, session):
        pass

def remove_solvent_ligands_ions(atoms, keep = None):
    '''Remove solvent, ligands and ions unless that removes all atoms
    in which case don't remove any.'''
    # TODO: Properly identify solvent, ligands and ions.
    # Currently simply remove every atom is does not belong to a chain.
    fatoms = atoms.filter(atoms.in_chains)
    if keep:
        fatoms = fatoms.merge(atoms.intersect(keep))
    all_small = (len(fatoms) == 0)
    if all_small:
        return atoms, all_small
    return fatoms, all_small

def surface_rgba(color, transparency, chain_id = None):
    if color is None:
        if chain_id is None:
            from numpy import array, uint8
            rgba8 = array((180,180,180,255), uint8)
        else:
            from .colors import chain_rgba8
            rgba8 = chain_rgba8(chain_id)
    else:
        rgba8 = color.uint8x4()
    if not transparency is None:
        opacity = int(255*(100.0-transparency)/100.0)
        rgba8[3] = opacity
    return rgba8

def update_color(surf, color, transparency):
    if color is None:
        if not transparency is None:
            opacity = int(255*(100.0-transparency)/100.0)
            vcolors = surf.vertex_colors
            if vcolors is None:
                rgba = surf.color
                rgba[3] = opacity
                surf.color = rgba
            else:
                vcolors[:,3] = opacity
                surf.vertex_colors = vcolors
    else:
        rgba = color.uint8x4()
        if not transparency is None:
            opacity = int(255*(100.0-transparency)/100.0)
            rgba[3] = opacity
        surf.color = rgba
        surf.vertex_colors = None

def surfaces_overlapping_atoms(surfs, atoms):
    si = atoms.intersects_each([s.atoms for s in surfs])
    osurfs = [s for s,i in zip(surfs,si) if i]
    return osurfs

def surfaces_with_atoms(atoms, models):
    surfs = []
    for m in list(atoms.unique_structures) + [models.drawing]:
        for s in m.child_drawings():
            if isinstance(s, MolecularSurface):
                if len(atoms.intersect(s.atoms)) > 0:
                    surfs.append(s)
    return surfs

def show_surface_atom_patches(atoms, models, only = False):
    surfs = surfaces_with_atoms(atoms, models)
    for s in surfs:
        s.show(atoms & s.atoms, only = only)
    return surfs

def show_surface_patches(surf_models, only = False):
    for s in surf_models:
        s.show(s.atoms, only = only)

def hide_surface_atom_patches(atoms, models):
    surfs = surfaces_with_atoms(atoms, models)
    for s in surfs:
        s.hide(atoms & s.atoms)
    return surfs

def hide_surface_patches(surf_models):
    for s in surf_models:
        s.hide(s.atoms)

def close_surfaces(atoms_or_surfs, models):
    from . import Atoms
    surfs = (surfaces_with_atoms(atoms_or_surfs, models)
             if isinstance(atoms_or_surfs, Atoms) else atoms_or_surfs)
    if surfs:
        models.close(surfs)

def buried_area(a1, a2, probe_radius):
    from ..surface import spheres_surface_area
    xyz1, r1 = atom_spheres(a1, probe_radius)
    a1a = spheres_surface_area(xyz1, r1)
    xyz2, r2 = atom_spheres(a2, probe_radius)
    a2a = spheres_surface_area(xyz2, r2)
    from numpy import concatenate
    xyz12, r12 = concatenate((xyz1,xyz2)), concatenate((r1,r2))
    a12a = spheres_surface_area(xyz12, r12)
    ba = 0.5 * (a1a.sum() + a2a.sum() - a12a.sum())
    return ba, a1a, a2a, a12a

def atom_spheres(atoms, probe_radius = 1.4):
    xyz = atoms.scene_coords
    r = atoms.radii.copy()
    r += probe_radius
    return xyz, r
