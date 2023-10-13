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
molsurf: Compute molecular surfaces
===================================
"""

from chimerax.core.models import Surface

# If MOLSURF_STATE_VERSION changes, then bump the bundle's
# (maximum) session version number.
MOLSURF_STATE_VERSION = 1

class MolecularSurface(Surface):
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
    update : bool
      Whether the surface automically updates its shape when atom coordinates change.
      Default true.
    '''

    def __init__(self, session, enclose_atoms, show_atoms, probe_radius, grid_spacing,
                 resolution, level, name, color, visible_patches, sharp_boundaries, update=True):
        
        Surface.__init__(self, name, session)
        self.selection_coupled = enclose_atoms.unique_structures

        self.atoms = enclose_atoms
        self._atom_count = len(self.atoms)	# Used to determine when atoms deleted.
        self.show_atoms = show_atoms	# Atoms for surface patch to show
        self.probe_radius = probe_radius # Only used for solvent excluded surface
        self.grid_spacing = grid_spacing
        self.resolution = resolution    # Only used for Gaussian surface
        self.level = level		# Contour level for Gaussian surface, atomic number units
        self.color = color
        self._atom_patch_colors = None
        self._atom_patch_color_mask = None
        self.visible_patches = visible_patches
        self.sharp_boundaries = sharp_boundaries
        self._joined_triangles = None
        self._refinement_steps = 1	# Used for fixing sharp edge problems near 3 atom junctions.

        self._auto_update_handler = None
        self.auto_update = update

        self._vertex_to_atom = None
        self._vertex_to_atom_count = None	# Used to check if atoms deleted
        self._max_radius = None
        self.clip_cap = True

    def delete(self):
        self.auto_update = False  # Remove auto update handler
        Surface.delete(self)
        
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
            self._clear_shape()
        elif shown_changed:
            self.triangle_mask = self._calc_triangle_mask()

    def _clear_shape(self):
        from numpy import empty, float32, int32
        va = na = empty((0,3),float32)
        ta = empty((0,3),int32) 
        self.set_geometry(va, na, ta)
        self.color = self._average_color()
        self.vertex_colors = None
        self._vertex_to_atom = None
        self._max_radius = None
        self._joined_triangles = None

    def _get_auto_update(self):
        return self._auto_update_handler is not None
    def _set_auto_update(self, enable):
        h = self._auto_update_handler
        if enable and h is None:
            from chimerax.atomic import get_triggers
            t = get_triggers()
            self._auto_update_handler = t.add_handler('changes', self._atoms_changed)
            if self.vertices is not None:
                self._recompute_shape()
        elif not enable and h:
            from chimerax.atomic import get_triggers
            t = get_triggers()
            t.remove_handler(h)
            self._auto_update_handler = None
    auto_update = property(_get_auto_update, _set_auto_update)

    def _atoms_changed(self, trigger_name, changes):
        if self.deleted:
            return 'delete handler'
        if self._coordinates_changed(changes):
            self._recompute_shape()

    def _coordinates_changed(self, changes):
        if 'active_coordset changed' in changes.structure_reasons():
            # Active coord set index changed.  Playing a trajectory.
            changed_structures = set(changes.modified_structures())
            for s in self.atoms.unique_structures:
                if s in changed_structures:
                    return True
        if 'coord changed' in changes.atom_reasons():
            # Atom coordinates changed through Atom or Atoms set_coord()
            if self.atoms.intersects(changes.modified_atoms()):
                return True
        elif 'coordset changed' in changes.coordset_reasons():
            # Atom coordinates changed through CoordSet object.
            # TODO: If 'coord changed' and 'coordset changed' currently we only
            #   look at 'coord changed' atoms to avoid updating surfaces for
            #   atoms that have not changed.  For example changing one rotamer
            #   only updates one chain surface instead of all chain surfaces.
            #   Currently there is no Python CoordSet method to set coordinates.
            changed_structures = set(cs.structure for cs in changes.modified_coordsets())
            for s in self.atoms.unique_structures:
                if s in changed_structures:
                    return True
        if changes.num_deleted_atoms() > 0 and len(self.atoms) < self._atom_count:
            return True
        return False

    def _recompute_shape(self):
        if len(self.atoms) == 0:
            self.session.models.close([self])
        else:
            self._clear_shape()
            self.calculate_surface_geometry()
        
    @property
    def atom_count(self):
        '''Number of atoms for calculating the surface. Read only.'''
        return len(self.atoms)

    def atom_coords(self):
        atoms = self.atoms
        return atoms.coords if atoms.single_structure else atoms.scene_coords
    
    def calculate_surface_geometry(self):
        '''Recalculate the surface if parameters have been changed.'''
        if self.vertices is not None and len(self.vertices) > 0:
            return              # Geometry already computed

        atoms = self.atoms
        xyz = self.atom_coords()
        res = self.resolution
        from chimerax import surface
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
            from chimerax.surface import sharp_edge_patches
            va, na, ta, tj, v2a = sharp_edge_patches(va, na, ta, v2a, xyz, **kw)
            self._joined_triangles = tj	# With non-duplicate vertices for clip cap calculation
            self._vertex_to_atom = v2a

        self.set_geometry(va, na, ta)
        self.triangle_mask = self._calc_triangle_mask()
        self._show_atom_patch_colors()
        self.update_selection()
        self._atom_count = len(atoms)

    def _calc_triangle_mask(self):
        tmask = self._patch_display_mask(self.show_atoms)
        if self.visible_patches is None:
            return tmask

        from chimerax import surface
        m = surface.largest_blobs_triangle_mask(self.vertices, self.joined_triangles, tmask,
                                                blob_count = self.visible_patches,
                                                rank_metric = 'area rank')
        return m

    def _get_joined_triangles(self):
        if self.sharp_boundaries:
            tri = self._joined_triangles
            if tri is None:
                from chimerax import surface
                # With sharp boundaries triangles are not connected.
                vmap = surface.unique_vertex_map(self.vertices)
                self._joined_triangles = tri = vmap[self.triangles]
        else:
            tri = self.triangles
        return tri
    def _set_joined_triangles(self, jtri):
        self._joined_triangles = jtri
    joined_triangles = property(_get_joined_triangles, _set_joined_triangles)
    
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
            from chimerax import geometry
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
        elif self._vertex_to_atom is not None:
            if len(self.atoms) < self._vertex_to_atom_count:
                # Atoms deleted
                self._vertex_to_atom = None
            elif len(self._vertex_to_atom) != len(self.vertices):
                # Some other code like color zone with sharp_edges = True
                # changed the surface geometery.
                self._vertex_to_atom = None
        return self._vertex_to_atom

    def _vertices_for_atoms(self, atoms):
        if atoms is None:
            nv = len(self.vertices)
            v = slice(nv)
            all_atoms = True
        else:
            ai = self.atoms.mask(atoms)
            v2a = self.vertex_to_atom_map()
            all_atoms = ai.all()
            if all_atoms:
                nv = len(self.vertices)
                v = slice(nv)
            elif v2a is not None:
                v = ai[v2a]		# Vertices for the given atoms
            else:
                v = None
        return v, all_atoms

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
        from numpy import logical_and, empty
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

    def _get_model_color(self):
        vc = self.vertex_colors
        from chimerax.core.colors import most_common_color
        return self.color if vc is None else most_common_color(vc)
    def _set_model_color(self, color):
        self.color = color
        self.vertex_colors = None
        self._atom_patch_colors = None
        self._atom_patch_color_mask = None
    model_color = property(_get_model_color, _set_model_color)

    def _average_color(self):
        vc = self.vertex_colors
        if vc is None or len(vc) == 0:
            return self.color
        from numpy import float32, uint8
        csum = vc.sum(axis = 0, dtype = float32)
        csum /= len(vc)
        acolor = csum.astype(uint8)
        return acolor

    def color_atom_patches(self, atoms = None,
                           color = None, per_atom_colors = None, vertex_colors = None,
                           opacity = None):
        '''
        Set surface patch colors for specified atoms.  The atoms collection may include atoms that
        are not included in the surface. If atoms is None then all atom patches are colored.
        Colors can be specified in one of four ways.
        A single color can be specified.  Or an array of colors for the specified atoms
        per_atom_colors can be given.  Or an array of whole surface vertex colors can be given.
        If none of those color arguments are given, then the colors of the atoms are used.
        Transparency is set if opacity is given, otherwise transparency remains the same as the
        current surface transparency.  Return value is True if atom patches were colored but
        can be False if no coloring was done because this surface has no atom to vertex mapping
        and coloring was requested for only a subset of atoms.
        '''
        v, all_atoms = self._vertices_for_atoms(atoms)
        if v is None:
            # No atom to vertex mapping and only some atoms are being colored.
            return False

        if all_atoms:
            # Optimize the common case of all atoms being colored.
            if color is not None:
                self._set_color_and_opacity(color, opacity)
            elif vertex_colors is not None:
                if opacity is not None:
                    vertex_colors[:,3] = opacity
                self.vertex_colors = vertex_colors
                self._clear_atom_patch_colors()
            else:
                atom_colors, vertex_colors = self._per_atom_colors(atoms, per_atom_colors,
                                                                   opacity=opacity)
                self.vertex_colors = vertex_colors
                self._remember_atom_patch_colors(atom_colors)
        else:
            # Subset of atoms are being colored.
            vcolors = self.get_vertex_colors(create = True, copy = True)
            if color is not None:
                vcolors[v,:3] = color[:3]
            elif vertex_colors is not None:
                vcolors[v,:3] = vertex_colors[v,:3]
            else:
                atom_colors, vc = self._per_atom_colors(atoms, per_atom_colors)
                vcolors[v,:3] = vc[v,:3]
            if opacity is not None:
                vcolors[v,3] = opacity
            self.vertex_colors = vcolors
            self._update_atom_patch_colors(atoms, color, per_atom_colors, vertex_colors)

        return True
    
    def _per_atom_colors(self, atoms, per_atom_colors, opacity = None):
        if per_atom_colors is None:
            atom_colors = self.atoms.colors
        else:
            atom_colors = per_atom_colors[atoms.indices(self.atoms),:]
        if opacity is not None:
            atom_colors[:,3] = opacity
        v2a = self.vertex_to_atom_map()
        if v2a is None:
            from chimerax.core.errors import UserError
            raise UserError('Surface #%s does not have atom patches, cannot color by atom'
                            % self.id_string)
        vertex_colors = atom_colors[v2a,:]
        return atom_colors, vertex_colors

    def _remember_atom_patch_colors(self, atom_colors):
        self._atom_patch_colors = atom_colors
        from numpy import ones
        self._atom_patch_color_mask = ones((len(atom_colors),), bool)

    def _update_atom_patch_colors(self, atoms, color, per_atom_colors, vertex_colors):
        if vertex_colors is not None:
            m = self._atom_patch_color_mask
            if m is not None and len(m) == len(self.atoms):
                from numpy import putmask
                putmask(m, self.atoms.mask(atom), 0)
            return
        if color is not None:
            acolors = color
        elif per_atom_colors is not None:
            acolors = per_atom_colors[atoms.mask(self.atoms)]
        else:
            acolors = atoms.intersect(self.atoms).colors
        ai = self.atoms.mask(atoms)
        apc = self._atom_patch_colors
        if apc is None or len(apc) != len(self.atoms):
            na = len(self.atoms)
            from numpy import empty, uint8
            self._atom_patch_colors = c = empty((na,4), uint8)
            c[ai] = acolors
            self._atom_patch_color_mask = ai
        else:
            apc[ai] = acolors
            m = self._atom_patch_color_mask
            from numpy import logical_or
            logical_or(m, ai, m)
    
    def _clear_atom_patch_colors(self):
        self._atom_patch_colors = None
        self._atom_patch_color_mask = None

    def _show_atom_patch_colors(self):
        apc = self._atom_patch_colors
        if apc is None or len(apc) != len(self.atoms):
            return
        m = self._atom_patch_color_mask
        self.color_atom_patches(self.atoms.filter(m), per_atom_colors = apc[m])
            
    def _set_color_and_opacity(self, color = None, opacity = None):
        c8 = self.color if color is None else color
        if opacity is not None:
            c8 = (c8[0], c8[1], c8[2], opacity)
        self.model_color = c8
        self._clear_atom_patch_colors()

    # Handle undo of color changes
    _color_attributes = ('color', 'vertex_colors',
                         '_atom_patch_colors', '_atom_patch_color_mask')
    def _color_undo_state(self):
        color_state = {}
        from numpy import ndarray
        for attr in self._color_attributes:
            value = getattr(self, attr)
            if isinstance(value, ndarray):
                value = value.copy()
            color_state[attr] = value
        return color_state
    def _restore_colors_from_undo_state(self, color_state):
        for attr in self._color_attributes:
            setattr(self, attr, color_state[attr])
    color_undo_state = property(_color_undo_state, _restore_colors_from_undo_state)
    
    def first_intercept(self, mxyz1, mxyz2, exclude = None):
        # Pick atom associated with surface patch
        from chimerax.graphics import Drawing, PickedTriangle
        p = Drawing.first_intercept(self, mxyz1, mxyz2, exclude)
        if not isinstance(p, PickedTriangle) or p.drawing() is not self:
            return p
        t = p.triangle_number
        v = self.triangles[t,0]
        v2a = self.vertex_to_atom_map()
        if v2a is None:
            from chimerax.core.models import PickedModel
            pa = PickedModel(self, p.distance)
        else:
            a = v2a[v]
            atom = self.atoms[a]
            from .structure import PickedAtom
            pa = PickedAtom(atom, p.distance)
        if isinstance(p, PickedTriangle):
            pa.triangle_pick = p	# Used by for reporting surface color value
        return pa

    def set_selected(self, sel, *, fire_trigger=True):
        self.atoms.selected = sel
        self.update_selection(fire_trigger=fire_trigger)
        if not self.has_atom_patches():
            Surface.set_selected(self, sel, fire_trigger=fire_trigger)
    selected = property(Surface.selected.fget, set_selected)

    def update_selection(self, *, fire_trigger=True):
        asel = self.atoms.selected
        tmask = self._atom_triangle_mask(asel)
        if tmask is None:
            if not self.has_atom_patches():
                return
            sel_val = False
        else:
            sel_val = (tmask.sum() > 0)
        Surface.set_selected(self, sel_val, fire_trigger=fire_trigger)
        self.highlighted_triangles_mask = tmask

    # State save/restore in ChimeraX
    _save_attrs = ('_refinement_steps', '_vertex_to_atom', '_vertex_to_atom_count', '_max_radius',
                   'vertices', 'normals', 'triangles', 'triangle_mask', 'vertex_colors', 'color',
                   '_joined_triangles', '_atom_patch_colors', '_atom_patch_color_mask')

    def take_snapshot(self, session, flags):
        init_attrs = ('atoms', 'show_atoms', 'probe_radius', 'grid_spacing', 'resolution', 'level',
                      'name', 'color', 'visible_patches', 'sharp_boundaries')
        data = {attr:getattr(self, attr) for attr in init_attrs}
        data['model state'] = Surface.take_snapshot(self, session, flags)
        data.update({attr:getattr(self,attr) for attr in self._save_attrs if hasattr(self,attr)})
        data['version'] = MOLSURF_STATE_VERSION
        return data

    @staticmethod
    def restore_snapshot(session, data):
        d = data
        s = MolecularSurface(session, d['atoms'], d['show_atoms'],
                             d['probe_radius'], d['grid_spacing'], d['resolution'],
                             d['level'], d['name'], d['color'], d['visible_patches'],
                             d['sharp_boundaries'])
        Surface.set_state_from_snapshot(s, session, d['model state'])
        geom_attrs = ('vertices', 'normals', 'triangles')
        s.set_geometry(d['vertices'], d['normals'], d['triangles'])
        for attr in MolecularSurface._save_attrs:
            if attr in d and attr not in geom_attrs:
                setattr(s, attr, d[attr])
        return s

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

def surface_initial_color(color, transparency, atoms = None):
    if color is None:
        if atoms is None:
            from numpy import array, uint8
            rgba8 = array((180,180,180,255), uint8)
        else:
            rgba8 = atoms.average_ribbon_color
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

def surfaces_with_atoms(atoms):
    if atoms is None or len(atoms) == 0:
        return []
    top_drawing = atoms[0].structure.session.models.scene_root_model

    surfs = []
    for m in list(atoms.unique_structures) + [top_drawing]:
        for s in m.child_drawings():
            if isinstance(s, MolecularSurface):
                if s.atoms.intersects(atoms):
                    surfs.append(s)
    return surfs

def show_surface_atom_patches(atoms, only = False):
    surfs = [s for s in surfaces_with_atoms(atoms) if s.has_atom_patches()]
    for s in surfs:
        s.show(atoms & s.atoms, only = only)
    return surfs

def show_surface_patches(surf_models, only = False):
    for s in surf_models:
        s.show(s.atoms, only = only)

def hide_surface_atom_patches(atoms):
    surfs = [s for s in surfaces_with_atoms(atoms) if s.has_atom_patches()]
    for s in surfs:
        s.hide(atoms & s.atoms)
    return surfs

def hide_surface_patches(surf_models):
    for s in surf_models:
        s.hide(s.atoms)

def close_surfaces(atoms_or_surfs):
    from . import Atoms
    surfs = (surfaces_with_atoms(atoms_or_surfs)
             if isinstance(atoms_or_surfs, Atoms) else atoms_or_surfs)
    if surfs:
        models = surfs[0].session.models
        models.close(surfs)

def buried_area(a1, a2, probe_radius):
    from chimerax.surface import spheres_surface_area
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
