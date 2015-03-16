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
               ('chains', cli.BoolArg)])

def surface_command(session, atoms = None, probeRadius = 1.4, gridSpacing = 0.5,
                    color = None, transparency = 0, chains = False):
    '''
    Compute and display a solvent excluded molecular surface for each molecule.
    '''
    surfs = []
    for name, xyz, r, place in atom_spec_spheres(atoms,session,chains):
        from . import surface
        va,na,ta = surface.ses_surface_geometry(xyz, r, probeRadius, gridSpacing)
        # Create surface model to show surface
        name = '%s SES surface' % name
        rgba = surface_rgba(color, transparency, chains, name)
        surf = show_surface(name, va, na, ta, rgba, place)
        session.models.add([surf])
        surfs.append(surf)
    return surfs

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

def surface_rgba(color, transparency, chains, rand_seed):
    from .color import Color
    if chains and color is None:
        from random import uniform, seed
        seed(rand_seed)
        color = Color((uniform(.5,1),uniform(.5,1),uniform(.5,1),1))
    if color is None:
        color = Color((.7,.7,.7,1))
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

    ba = buried_area(a1, a2, probeRadius)

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

def atom_spheres(atoms, probe_radius = 1.4):
    xyz = atoms.coords
    r = atoms.radii.copy()
    r += probe_radius
    return xyz, r

def register_buriedarea_command():
    cli.register('buriedarea', _buriedarea_desc, buriedarea_command)
    cli.register('shake', _shake_desc, shake_command)

_shake_desc = cli.CmdDesc(
    required = [('atoms', atomspec.AtomSpecArg),],
    keyword = [('probeRadius', cli.FloatArg),])

def shake_command(session, atoms, probeRadius = 1.4):
    '''
    Compute solvent accessible surface area.
    Only the specified atoms are considered.
    '''
    a = atoms.evaluate(session).atoms
    xyz, r = atom_spheres(a, probeRadius)
    s = [(cid,xyz[cmask],r[cmask]) for cid, cmask in chain_indices(a)]
    ba = buried_areas(s)

    # Report result
    msg = '%d buried areas: ' % len(ba) + ', '.join('%s %s %.0f' % a for a in ba)
    log = session.logger
    log.info(msg)
    log.status(msg)

def buried_areas(s, min_area = 1):
    areas = []
    from .surface import spheres_surface_area
    for name, xyz, r in s:
        areas.append(spheres_surface_area(xyz, r).sum())

    buried = []
    n = len(s)
    for i in range(n):
        n1, xyz1, r1 = s[i]
        for j in range(i+1,n):
            n2, xyz2, r2 = s[j]
            from numpy import concatenate
            xyz12, r12 = concatenate((xyz1,xyz2)), concatenate((r1,r2))
            a12 = spheres_surface_area(xyz12, r12).sum()
            ba = 0.5 * (areas[i] + areas[j] - a12)
            if ba >= min_area:
                buried.append((n1, n2, ba))
    buried.sort(key = lambda a: a[2], reverse = True)

    return buried
