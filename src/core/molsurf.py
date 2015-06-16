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
                    color = None, transparency = 0, nthread = None):
    '''
    Compute and display solvent excluded molecular surfaces.
    '''
    atoms = check_atoms(atoms, session)
    pieces = [SurfCalc(m, cid, catoms, probe_radius, grid_spacing)
              for m, cid, catoms in atoms.by_chain]

    # Compute surfaces using multiple threads
    args = [(p,) for p in pieces]
    args.sort(key = lambda p: p[0].atom_count, reverse = True)      # Largest first for load balancing
    from . import threadq
    threadq.apply_to_list(lambda p: p.calculate_surface_geometry(), args, nthread)
#    for p in pieces:
#        p.calculate_surface_geometry()

    # Creates surface models
    surfs = []
    for p in pieces:
        # Create surface model to show surface
        sname = '%s_%s SES surface' % (p.mol.name, p.chain_id)
        rgba = surface_rgba(color, transparency, p.chain_id)
        surf = show_surface(sname, p.vertices, p.normals, p.triangles, rgba)
#        vatom = p.vertex_to_atom_map()
#        surf.vertex_colors = p.catoms.colors[vatom,:].copy()
        session.models.add([surf], parent = p.mol)
        surfs.append(surf)

    for p in pieces:
        if len(p.patoms) < len(p.catoms):
            print ('show patch', len(p.patoms), len(p.catoms))
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
                   ('nthread', cli.IntArg)],
        synopsis = 'create molecular surface')
    cli.register('surface', _surface_desc, surface_command)

def check_atoms(atoms, session):
    if atoms is None:
        from .structure import all_atoms
        atoms = all_atoms(session)
        if len(atoms) == 0:
            raise cli.AnnotationError('No atomic models open.')
    elif len(atoms) == 0:
        raise cli.AnnotationError('No atoms specified by %s' % (atoms.spec,))
    return atoms

class SurfCalc:

    def __init__(self, mol, chain_id, atoms, probe_radius, grid_spacing):
        self.mol = mol
        self.chain_id = chain_id
        self.patoms = self.remove_solvent_ligands_ions(atoms)	# Atoms for surface patch to show
        catoms = self.chain_atoms(mol, chain_id)		# Full chain atoms
        self.catoms = self.remove_solvent_ligands_ions(catoms)
        self.probe_radius = probe_radius
        self.grid_spacing = grid_spacing

    @property
    def atom_count(self):
        return len(self.catoms)

    def calculate_surface_geometry(self):
        atoms = self.catoms
        xyz = atoms.coords
        r = atoms.radii
        from .surface import ses_surface_geometry
        va, na, ta = ses_surface_geometry(xyz, r, self.probe_radius, self.grid_spacing)
        self.vertices = va
        self.normals = na
        self.triangles = ta

    # Chain atoms with solvent, ligands and ions filtered out.
    def chain_atoms(self, mol, chain_id):
        atoms = mol.atoms
        return atoms.filter(atoms.chain_ids == chain_id)

    def remove_solvent_ligands_ions(self, atoms):
        # TODO: Remove ligands and ions
        solvent = atoms.filter(atoms.residues.names == 'HOH')
        return atoms.subtract(solvent) if len(solvent) > 0 else atoms

    def vertex_to_atom_map(self):
        xyz1 = self.vertices
        xyz2 = self.catoms.coords
        max_dist = 3
        from . import geometry
        i1, i2, nearest1 = geometry.find_closest_points(xyz1, xyz2, max_dist)
        from numpy import empty, int32
        v2a = empty((len(xyz1),), int32)
        v2a[i1] = nearest1
        return v2a

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

def buriedarea_command(session, atoms1, atoms2, probe_radius = 1.4):
    '''
    Compute solvent accessible surface area.
    Only the specified atoms are considered.
    '''
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
        required = [('atoms1', AtomsArg), ('atoms2', AtomsArg)],
        keyword = [('probe_radius', cli.FloatArg),],
        synopsis = 'compute buried area')
    cli.register('buriedarea', _buriedarea_desc, buriedarea_command)
