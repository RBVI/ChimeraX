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

def measure_sasa(session, atoms = None, probe_radius = 1.4, sum = None):
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

    # Report results
    area = areas.sum()
    msg = 'Solvent accessible area for %s = %.5g' % (atoms.spec, area)
    log = session.logger
    log.info(msg)
    if sum is not None:
        a = areas[atoms.mask(sum)]
        area = a.sum()
        msg = ('Solvent accessible area for %s (%d atoms) of %s = %.5g'
               % (sum.spec, len(a), atoms.spec, area))
        log.info(msg)
    log.status(msg)

def register_command(session):
    from . import CmdDesc, register, AtomsArg, FloatArg
    _sasa_desc = CmdDesc(
        optional = [('atoms', AtomsArg)],
        keyword = [('probe_radius', FloatArg),
                   ('sum', AtomsArg)],
        synopsis = 'compute solvent accessible surface area')
    register('measure sasa', _sasa_desc, measure_sasa)
