# vim: set expandtab shiftwidth=4 softtabstop=4:

def sasa(session, atoms = None, probe_radius = 1.4, sum = None):
    '''
    Compute solvent accessible surface area.

    Parameters
    ----------
    atoms : Atoms
      A probe sphere is rolled over these atoms ignoring collisions with any other atoms.
    probe_radius : float
      Radius of the probe sphere.
    sum : Atoms
      Sum the accessible areas per atom only over these atoms.
    '''
    from .surface import check_atoms
    atoms = check_atoms(atoms, session)
    r = atoms.radii
    r += probe_radius
    from ..surface import spheres_surface_area
    areas = spheres_surface_area(atoms.scene_coords, r)
    if sum is None:
        area = areas.sum()
        msg = 'Solvent accessible area for %s = %.5g' % (atoms.spec, area)
    else:
        a = areas[atoms.mask(sum)]
        area = a.sum()
        msg = ('Solvent accessible area for %s (%d atoms) of %s = %.5g'
               % (sum.spec, len(a), atoms.spec, area))
    log = session.logger
    log.info(msg)
    log.status(msg)

def register_command(session):
    from . import CmdDesc, register, AtomsArg, FloatArg
    _sasa_desc = CmdDesc(
        optional = [('atoms', AtomsArg)],
        keyword = [('probe_radius', FloatArg),
                   ('sum', AtomsArg)],
        synopsis = 'compute solvent accessible surface area')
    register('sasa', _sasa_desc, sasa)
