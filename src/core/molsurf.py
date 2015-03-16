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
               ('transparency', cli.FloatArg),])

def surface_command(session, atoms = None, probeRadius = 1.4, gridSpacing = 0.5,
                    color = color.Color((.7,.7,.7,1)), transparency = 0):
    '''
    Compute and display a solvent excluded molecular surface for each molecule.
    '''
    surfs = []
    for name, a, place in atom_blobs(atoms,session):
        xyz = a.coords
        r = a.radii
        from . import surface
        va,na,ta = surface.ses_surface_geometry(xyz, r, probeRadius, gridSpacing)

        # Create surface model to show surface
        name = '%s SES surface' % name
        rgba8 = color.uint8x4()
        rgba8[3] = int(rgba8[3] * (100.0-transparency)/100.0)
        surf = show_surface(name, va, na, ta, rgba8, place)
        session.models.add([surf])
        surfs.append(surf)
    return surfs

def atom_blobs(atom_spec, session):
    if atom_spec is None:
        from .structure import StructureModel
        ab = [(m.name, m.mol_blob.atoms, m.position)
              for m in session.models.list()
              if isinstance(m, StructureModel)]
    else:
        a = atom_spec.evaluate(session).atoms
        if a is None or len(a) == 0:
            raise cli.AnnotationError('No atoms specified by %s' % (str(atoms),))
        ab = [(str(atom_spec), a, None)]
    return ab

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
    for name, a, place in atom_blobs(atoms,session):
        xyz = a.coords
        r = a.radii.copy()
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

    # Calculate areas
    from .surface import spheres_surface_area
    xyz1, r1 = atom_spheres(a1, probeRadius)
    a1a = spheres_surface_area(xyz1, r1).sum()
    xyz2, r2 = atom_spheres(a2, probeRadius)
    a2a = spheres_surface_area(xyz2, r2).sum()
    from numpy import concatenate
    xyz12, r12 = concatenate((xyz1,xyz2)), concatenate((r1,r2))
    a12a = spheres_surface_area(xyz12, r12).sum()
    ba = 0.5 * (a1a + a2a - a12a)

    # Report result
    msg = 'Buried area between %s and %s = %.5g' % (str(atoms1), str(atoms2), ba)
    log = session.logger
    log.status(msg)
    msg += ('\n  area %s = %.5g, area %s = %.5g, area both = %.5g'
            % (str(atoms1), a1a, str(atoms2), a2a, a12a))
    log.info(msg)

def atom_spheres(atoms, probe_radius = 1.4):
    xyz = atoms.coords
    r = atoms.radii.copy()
    r += probe_radius
    return xyz, r

def register_buriedarea_command():
    cli.register('buriedarea', _buriedarea_desc, buriedarea_command)
