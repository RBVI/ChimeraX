# vi: set expandtab shiftwidth=4 softtabstop=4:
"""
molsurf -- Compute molecular surfaces
=====================================
"""

from . import generic3d

class MolecularSurface(generic3d.Generic3DModel):
    pass

from . import cli, atomspec, color
_surface_desc = cli.CmdDesc(
    optional = [('atoms', atomspec.AtomSpecArg)],
    keyword = [('probeRadius', cli.FloatArg),
               ('gridSpacing', cli.FloatArg),
               ('color', color.ColorArg),
               ('transparency', cli.FloatArg),
               ('chains', cli.BoolArg),
               ('nthread', cli.IntArg)])

def surface_command(session, atoms = None, probeRadius = 1.4, gridSpacing = 0.5,
                    color = None, transparency = 0, chains = False, nthread = None):
    '''
    Compute and display a solvent excluded molecular surface for each molecule.
    '''
    surfs = []

    # Compute surfaces using multiple threads
    spheres = atom_spec_spheres(atoms,session,chains)
    args = [(name,xyz,r,place,probeRadius,gridSpacing)
            for name, xyz, r, place in spheres]
    args.sort(key = lambda a: len(a[1]), reverse = True)        # Largest first
    from . import threadq
    geom = threadq.apply_to_list(calc_surf, args, nthread)	# geom does not match args ordering

    # Creates surface models
    for name, place, va, na, ta in geom:
        # Create surface model to show surface
        sname = '%s SES surface' % name
        rgba = surface_rgba(color, transparency, chains, name)
        surf = show_surface(sname, va, na, ta, rgba, place)
        session.models.add([surf])
        surfs.append(surf)
    return surfs

def calc_surf(name, xyz, r, place, probe_radius, grid_spacing):
    from .surface import ses_surface_geometry
    va, na, ta = ses_surface_geometry(xyz, r, probe_radius, grid_spacing)
    return (name, place, va, na, ta)

def atom_spec_spheres(atom_spec, session, chains = False):
    if atom_spec is None:
        s = []
        from .structure import StructureModel
        for m in session.models.list():
            if isinstance(m, StructureModel):
                a = m.mol_blob.atoms
                if chains:
                    for cname, ci in chain_indices(a):
                        xyz, r = a.coords, a.radii
                        s.append(('%s/%s'%(m.name,cname), xyz[ci], r[ci], m.position))
                else:
                    s.append((m.name, a.coords, a.radii, m.position))
    else:
        a = atom_spec.evaluate(session).atoms
        if a is None or len(a) == 0:
            raise cli.AnnotationError('No atoms specified by %s' % (str(atom_spec),))
        if chains:
            s = []
            for cname, ci in chain_indices(a):
                xyz, r = a.coords, a.radii
                s.append(('%s/%s'%(str(atom_spec),cname), xyz[ci], r[ci], None))
        else:
            s = [(str(atom_spec), a.coords, a.radii, None)]
        # TODO: Use correct position matrix for atoms
    return s

def chain_indices(atoms):
    import numpy
    atom_cids = numpy.array(atoms.residues.chain_ids)
    cids = numpy.unique(atom_cids)
    cid_masks = [(cid,(atom_cids == cid)) for cid in cids]
    return cid_masks

def surface_rgba(color, transparency, chains, cid):
    if color is None:
        if chains:
            from . import structure
            rgba8 = structure.chain_rgba8(cid.split('/',1)[-1])
        else:
            from numpy import array, uint8
            rgba8 = array((180,180,180,255), uint8)
    else:
        rgba8 = color.uint8x4()
        rgba8[3] = int(rgba8[3] * (100.0-transparency)/100.0)
    return rgba8

def show_surface(name, va, na, ta, color = (180,180,180,255), place = None):

    surf = MolecularSurface(name)
    if not place is None:
        surf.position = place
    surf.geometry = va, ta
    surf.normals = na
    surf.color = color
    return surf

def molecule_surface(mol, probe_radius = 1.4, grid_spacing = 0.5):
    a = mol.atoms(exclude_water = True)
    xyz, r = a.coords, a.radii
    from .surface import ses_surface_geometry
    va, na, ta = ses_surface_geometry(xyz, r, probe_radius, grid_spacing)
    from numpy import array, uint8
    color = array((180,180,180,255), uint8)
    surf = show_surface(mol.name + ' surface', va, na, ta, color, mol.position)
    return surf

def register_surface_command():
    cli.register('surface', _surface_desc, surface_command)

_sasa_desc = cli.CmdDesc(
    optional = [('atoms', atomspec.AtomSpecArg)],
    keyword = [('probeRadius', cli.FloatArg),])

def sasa_command(session, atoms = None, probeRadius = 1.4):
    '''
    Compute solvent accessible surface area.
    Only the specified atoms are considered.
    '''
    log = session.logger
    for name, xyz, r, place in atom_spec_spheres(atoms,session):
        r += probeRadius
        from . import surface
        areas = surface.spheres_surface_area(xyz, r)
        area = areas.sum()
        msg = 'Solvent accessible area for %s = %.5g' % (name, area)
        log.info(msg)
        log.status(msg)

def register_sasa_command():
    cli.register('sasa', _sasa_desc, sasa_command)

_buriedarea_desc = cli.CmdDesc(
    required = [('atoms1', atomspec.AtomSpecArg), ('atoms2', atomspec.AtomSpecArg)],
    keyword = [('probeRadius', cli.FloatArg),])

def buriedarea_command(session, atoms1, atoms2, probeRadius = 1.4):
    '''
    Compute solvent accessible surface area.
    Only the specified atoms are considered.
    '''
    a1 = atoms1.evaluate(session).atoms
    a2 = atoms2.evaluate(session).atoms
    ni = len(a1.intersect(a2))
    if ni > 0:
        raise cli.AnnotationError('Two sets of atoms must be disjoint, got %d atoms in %s and %s'
                                  % (ni, str(atoms1), str(atoms2)))

    ba, a1a, a2a, a12a = buried_area(a1, a2, probeRadius)

    # Report result
    msg = 'Buried area between %s and %s = %.5g' % (str(atoms1), str(atoms2), ba)
    log = session.logger
    log.status(msg)
    msg += ('\n  area %s = %.5g, area %s = %.5g, area both = %.5g'
            % (str(atoms1), a1a, str(atoms2), a2a, a12a))
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
    cli.register('buriedarea', _buriedarea_desc, buriedarea_command)
