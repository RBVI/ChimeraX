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

def measure_sasa(session, atoms = None, probe_radius = 1.4, sum = None,
                 set_attribute = True):
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
    set_attribute : bool
      Whether to set atom.area and residue.area values.
    '''
    from .surfacecmds import check_atoms
    atoms = check_atoms(atoms, session)
    r = atoms.radii
    r += probe_radius
    from . import spheres_surface_area
    areas = spheres_surface_area(atoms.scene_coords, r)

    # Set area atom and residue attributes
    if set_attribute:
        set_area_attributes(atoms, areas)
            
    # Report results
    area = areas.sum()
    aname = f'{atoms.spec} ({len(atoms)} atoms)' if hasattr(atoms, 'spec') else f'{len(atoms)} atoms'
    msg = 'Solvent accessible area for %s = %.5g' % (aname, area)
    log = session.logger
    log.info(msg)
    if sum is not None:
        a = areas[atoms.mask(sum)]
        sarea = a.sum()
        sname = f'{sum.spec} ({len(sum)} atoms)' if hasattr(sum, 'spec') else f'{len(sum)} atoms'
        msg = ('Solvent accessible area for %s of %s = %.5g' % (sname, aname, sarea))
        log.info(msg)
    log.status(msg)

    return area if sum is None else (area, sarea)

def set_area_attributes(atoms, areas):
    for a, area in zip(atoms, areas):
        a.area = area
    res = atoms.unique_residues
    for r in res:
        r.area = 0
    for a, area in zip(atoms, areas):
        a.residue.area += area
    if len(atoms) > 0:
        session = atoms[0].structure.session
        from chimerax.atomic import Atom, Residue
        Atom.register_attr(session, "area", "Measure SASA", attr_type=float)
        Residue.register_attr(session, "area", "Measure SASA", attr_type=float)
    
def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, FloatArg, BoolArg
    from chimerax.atomic import AtomsArg
    _sasa_desc = CmdDesc(
        optional = [('atoms', AtomsArg)],
        keyword = [('probe_radius', FloatArg),
                   ('sum', AtomsArg),
                   ('set_attribute', BoolArg)],
        synopsis = 'compute solvent accessible surface area')
    register('measure sasa', _sasa_desc, measure_sasa, logger=logger)
