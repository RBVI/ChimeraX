# vi: set expandtab shiftwidth=4 softtabstop=4:

def sasa_command(session, atoms = None, probe_radius = 1.4):
    '''
    Compute solvent accessible surface area.
    Only the specified atoms are considered.
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

def register_sasa_command():
    from . import CmdDesc, register, AtomsArg, FloatArg
    _sasa_desc = CmdDesc(
        optional = [('atoms', AtomsArg)],
        keyword = [('probe_radius', FloatArg),],
        synopsis = 'compute solvent accessible surface area')
    register('sasa', _sasa_desc, sasa_command)
