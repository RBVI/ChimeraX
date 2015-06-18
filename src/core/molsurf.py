# vi: set expandtab shiftwidth=4 softtabstop=4:
"""
molsurf -- Compute molecular surfaces
=====================================
"""

from . import generic3d
from . import cli

class MolecularSurface(generic3d.Generic3DModel):
    pass

def surface_command(session, atoms = None, enclose = None,
                    probe_radius = 1.4, grid_spacing = 0.5,
                    color = None, transparency = 0, nthread = None,
                    close = False, hide = False):
    '''
    Compute and display solvent excluded molecular surfaces.
    '''
    atoms = check_atoms(atoms, session) # Warn if no atoms specifed

    pieces = []
    if enclose is None:
        for m, chain_id, show_atoms in atoms.by_chain:
            matoms = m.atoms
            enclose_atoms = matoms.filter(matoms.chain_ids == chain_id)
            name = '%s_%s SES surface' % (m.name, chain_id)
            rgba = surface_rgba(color, transparency, chain_id)
            s = SurfCalc(enclose_atoms, show_atoms, probe_radius, grid_spacing, m, name, rgba)
            pieces.append(s)
    else:
        show_atoms = enclose if atoms is None else atoms
        mols = enclose.unique_molecules
        parent = mols[0] if len(mols) == 1 else session.models.drawing
        name = 'Surface %s' % enclose.spec
        rgba = (170,170,170,255) if color is None else color.uint8x4()
        s = SurfCalc(enclose, show_atoms, probe_radius, grid_spacing, parent, name, rgba)
        pieces.append(s)

    # Compute surfaces using multiple threads
    args = [(p,) for p in pieces]
    args.sort(key = lambda p: p[0].atom_count, reverse = True)      # Largest first for load balancing
    from . import threadq
    threadq.apply_to_list(lambda p: p.calculate_surface_geometry(), args, nthread)

    # Creates surface models
    surfs = [p.create_surface(session) for p in pieces]

    if close:
        close_surfaces(atoms, session.models)
        surfs = []

    if hide:
        hide_surfaces(atoms, session.models)

    return surfs

def register_surface_command():
    from .structure import AtomsArg
    from . import cli, color
    _surface_desc = cli.CmdDesc(
        optional = [('atoms', AtomsArg)],
        keyword = [('enclose', AtomsArg),
                   ('probe_radius', cli.FloatArg),
                   ('grid_spacing', cli.FloatArg),
                   ('color', color.ColorArg),
                   ('transparency', cli.FloatArg),
                   ('nthread', cli.IntArg),
                   ('close', cli.NoArg),
                   ('hide', cli.NoArg)],
        synopsis = 'create molecular surface')
    cli.register('surface', _surface_desc, surface_command)

def check_atoms(atoms, session):
    if atoms is None:
        from .structure import all_atoms
        atoms = all_atoms(session)
        if len(atoms) == 0:
            raise cli.AnnotationError('No atomic models open.')
        atoms.spec = 'all atoms'
    elif len(atoms) == 0:
        raise cli.AnnotationError('No atoms specified by %s' % (atoms.spec,))
    return atoms

class SurfCalc:

    def __init__(self, enclose_atoms, show_atoms, probe_radius, grid_spacing, parent_drawing, name, color):
        self.atoms = self.remove_solvent_ligands_ions(enclose_atoms)
        self.show_atoms = self.remove_solvent_ligands_ions(show_atoms)	# Atoms for surface patch to show
        self.probe_radius = probe_radius
        self.grid_spacing = grid_spacing
        self.parent_drawing = parent_drawing
        self.name = name
        self.color = color
        self.vertices = None
        self.normals = None
        self.triangles = None
        self.surface_model = self.find_matching_surface()
        self._vertex_to_atom = None

    def find_matching_surface(self):
        d = self.parent_drawing
        for s in d.child_drawings():
            if isinstance(s, MolecularSurface):
                if len(s.atoms) == self.atom_count and s.atoms == self.atoms:
                    return s
        return None
                
    @property
    def atom_count(self):
        return len(self.atoms)

    def calculate_surface_geometry(self):
        if not self.vertices is None:
            return              # Geometry already computed
        s = self.surface_model
        if s and s.probe_radius == self.probe_radius and s.grid_spacing == self.grid_spacing:
            return
        atoms = self.atoms
        xyz = atoms.coords
        r = atoms.radii
        from .surface import ses_surface_geometry
        va, na, ta = ses_surface_geometry(xyz, r, self.probe_radius, self.grid_spacing)
        self.vertices = va
        self.normals = na
        self.triangles = ta

    def create_surface(self, session):
        surf = self.surface_model
        if surf:
            surf.display = True
        if self.vertices is None:
            # Show new patch of existing surface
            surf.triangle_mask = surf._calc_surf.patch_display_mask(self.show_atoms)
        else:
            # Update existing surface geometry or create new surface.
            new_surf = surf is None
            if new_surf:
                surf = MolecularSurface(self.name)
            surf.geometry = self.vertices, self.triangles
            surf.normals = self.normals
            surf.triangle_mask = self.patch_display_mask(self.show_atoms)
            surf.color = self.color
            surf.atoms = self.atoms
            surf.probe_radius = self.probe_radius
            surf.grid_spacing = self.grid_spacing
            surf._calc_surf = self
            if new_surf:
                session.models.add([surf], parent = self.parent_drawing)
#        surf.vertex_colors = c.vertex_atom_colors()
        return surf

    def remove_solvent_ligands_ions(self, atoms):
        '''Remove solvent, ligands and ions unless that removes all atoms
        in which case don't remove any.'''
        # TODO: Properly identify solvent, ligands and ions.
        # Currently simply remove every atom is does not belong to a chain.
        # TODO: The following crashes, bug #105
#        fatoms = atoms.filter(atoms.in_chains)
        solvent = atoms.filter(atoms.residues.names == 'HOH')
        fatoms = atoms.subtract(solvent) if len(solvent) > 0 else atoms
        if len(fatoms) == 0:
            return atoms
        return fatoms

    def vertex_to_atom_map(self):
        v2a = self._vertex_to_atom
        if v2a is None:
            xyz1 = self.vertices
            xyz2 = self.atoms.coords
            max_dist = 3
            from . import geometry
            i1, i2, nearest1 = geometry.find_closest_points(xyz1, xyz2, max_dist)
            from numpy import empty, int32
            v2a = empty((len(xyz1),), int32)
            v2a[i1] = nearest1
            self._vertex_to_atom = v2a
        return v2a

    def vertex_atom_colors(self):
        vatom = self.vertex_to_atom_map()
        return self.atoms.colors[vatom,:]

    def patch_display_mask(self, patch_atoms):
        surf_atoms = self.atoms
        if len(patch_atoms) == len(surf_atoms):
            return None
        v2a = self.vertex_to_atom_map()
        shown_atoms = surf_atoms.mask(patch_atoms)
        shown_vertices = shown_atoms[v2a]
        t = self.triangles
        from numpy import logical_and, empty, bool
        shown_triangles = empty((len(t),), bool)
        logical_and(shown_vertices[t[:,0]], shown_vertices[t[:,1]], shown_triangles)
        logical_and(shown_triangles, shown_vertices[t[:,2]], shown_triangles)
        return shown_triangles

def surface_rgba(color, transparency, chain_id):
    if color is None:
        if chain_id is None:
            from numpy import array, uint8
            rgba8 = array((180,180,180,255), uint8)
        else:
            from . import structure
            rgba8 = structure.chain_rgba8(chain_id)
    else:
        rgba8 = color.uint8x4()
        rgba8[3] = int(rgba8[3] * (100.0-transparency)/100.0)
    return rgba8

def show_surface(name, va, na, ta, color = (180,180,180,255)):
    surf = MolecularSurface(name)
    surf.geometry = va, ta
    surf.normals = na
    surf.color = color
    return surf

def molecule_surface(mol, probe_radius = 1.4, grid_spacing = 0.5):
    a = mol.atoms
    from numpy import array
    a = a.filter(array(a.residues.names) != 'HOH')     # exclude waters
    xyz, r = a.coords, a.radii
    from .surface import ses_surface_geometry
    va, na, ta = ses_surface_geometry(xyz, r, probe_radius, grid_spacing)
    from numpy import array, uint8
    color = array((180,180,180,255), uint8)
    surf = show_surface(mol.name + ' surface', va, na, ta, color, mol.position)
    return surf

def surfaces_with_atoms(atoms, models):
    surfs = []
    for m in list(atoms.unique_molecules) + [models.drawing]:
        for s in m.child_drawings():
            if isinstance(s, MolecularSurface):
                if len(atoms.intersect(s.atoms)) > 0:
                    surfs.append(s)
    return surfs

def hide_surfaces(atoms, models):
    for s in surfaces_with_atoms(atoms, models):
        s.display = False

def close_surfaces(atoms, models):
    surfs = surfaces_with_atoms(atoms, models)
    if surfs:
        models.close(surfs)

def sasa_command(session, atoms = None, probe_radius = 1.4):
    '''
    Compute solvent accessible surface area.
    Only the specified atoms are considered.
    '''
    atoms = check_atoms(atoms, session)
    r = atoms.radii
    r += probe_radius
    from . import surface
    areas = surface.spheres_surface_area(atoms.scene_coords, r)
    area = areas.sum()
    msg = 'Solvent accessible area for %s = %.5g' % (atoms.spec, area)
    log = session.logger
    log.info(msg)
    log.status(msg)

def register_sasa_command():
    from .structure import AtomsArg
    _sasa_desc = cli.CmdDesc(
        optional = [('atoms', AtomsArg)],
        keyword = [('probe_radius', cli.FloatArg),],
        synopsis = 'compute solvent accessible surface area')
    cli.register('sasa', _sasa_desc, sasa_command)

def buriedarea_command(session, atoms1, with_atoms2 = None, probe_radius = 1.4):
    '''
    Compute solvent accessible surface area.
    Only the specified atoms are considered.
    '''
    if with_atoms2 is None:
        raise cli.AnnotationError('Require "with" keyword: buriedarea #1 with #2')
    atoms2 = with_atoms2

    ni = len(atoms1.intersect(atoms2))
    if ni > 0:
        raise cli.AnnotationError('Two sets of atoms must be disjoint, got %d atoms in %s and %s'
                                  % (ni, atoms1.spec, atoms2.spec))

    ba, a1a, a2a, a12a = buried_area(atoms1, atoms2, probe_radius)

    # Report result
    msg = 'Buried area between %s and %s = %.5g' % (atoms1.spec, atoms2.spec, ba)
    log = session.logger
    log.status(msg)
    msg += ('\n  area %s = %.5g, area %s = %.5g, area both = %.5g'
            % (atoms1.spec, a1a, atoms2.spec, a2a, a12a))
    log.info(msg)

def buried_area(a1, a2, probe_radius):
    from .surface import spheres_surface_area
    xyz1, r1 = atom_spheres(a1, probe_radius)
    a1a = spheres_surface_area(xyz1, r1).sum()
    xyz2, r2 = atom_spheres(a2, probe_radius)
    a2a = spheres_surface_area(xyz2, r2).sum()
    from numpy import concatenate
    xyz12, r12 = concatenate((xyz1,xyz2)), concatenate((r1,r2))
    a12a = spheres_surface_area(xyz12, r12).sum()
    ba = 0.5 * (a1a + a2a - a12a)
    return ba, a1a, a2a, a12a

def atom_spheres(atoms, probe_radius = 1.4):
    xyz = atoms.coords
    r = atoms.radii.copy()
    r += probe_radius
    return xyz, r

def register_buriedarea_command():
    from .structure import AtomsArg
    _buriedarea_desc = cli.CmdDesc(
        required = [('atoms1', AtomsArg)],
        keyword = [('with_atoms2', AtomsArg),
                   ('probe_radius', cli.FloatArg),],
        synopsis = 'compute buried area')
    cli.register('buriedarea', _buriedarea_desc, buriedarea_command)
