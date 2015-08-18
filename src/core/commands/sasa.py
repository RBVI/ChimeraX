# vi: set expandtab shiftwidth=4 softtabstop=4:

def sasa(session, atoms = None, probe_radius = 1.4):
    '''
    Compute solvent accessible surface area.

    :param atoms: A probe sphere is rolled over these atoms ignoring collisions with any other atoms.
    :param probe_radius: Radius of the probe sphere.
    '''
    from .surface import check_atoms
    atoms = check_atoms(atoms, session)
    r = atoms.radii
    r += probe_radius
    from ..surface import spheres_surface_area
    areas = spheres_surface_area(atoms.scene_coords, r)
    area = areas.sum()
    msg = 'Solvent accessible area for %s = %.5g' % (atoms.spec, area)
    log = session.logger
    log.info(msg)
    log.status(msg)

def register_command(session):
    from . import CmdDesc, register, AtomsArg, FloatArg
    _sasa_desc = CmdDesc(
        optional = [('atoms', AtomsArg)],
        keyword = [('probe_radius', FloatArg),],
        synopsis = 'compute solvent accessible surface area')
    register('sasa', _sasa_desc, sasa)
