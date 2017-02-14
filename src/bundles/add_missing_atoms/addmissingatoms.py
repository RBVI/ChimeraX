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

#
# Add hydrogens and heavy atoms so that OpenMM molecular dynamics can be used on an
# atomic structure.  A new structure is created.
#
def addmissingatoms(session, structures, minimization_steps = 0, keep_waters = False):

    if len(structures) == 0:
        from chimerax.core.errors import UserError
        raise UserError('No structures specified')

    for m in structures:
        add_missing_atoms(session, m, minimization_steps, keep_waters)

def add_missing_atoms(session, m, minimization_steps = 0, keep_waters = False):
    fname = m.filename
    from pdbfixer import PDBFixer
    pf = PDBFixer(filename = fname)
    pf.findMissingResidues()
    pf.findNonstandardResidues()
    pf.replaceNonstandardResidues()
    pf.findMissingAtoms()
    pf.addMissingAtoms()
    pf.removeHeterogens(keep_waters)
    pf.addMissingHydrogens(7.0)
    if minimization_steps > 0:
        minimize(pf, minimization_steps)
    from os.path import splitext
    fout = splitext(fname)[0] + '-pdbfixer.pdb'
    out = open(fout, 'w')
    from simtk.openmm.app import PDBFile
    PDBFile.writeFile(pf.topology, pf.positions, out)
    out.close()
    mfix = session.models.open([fout])[0]
    mfix.atoms.displays = True
    mfix.residues.ribbon_displays = False
    m.display = False
    log = session.logger
    log.info('Wrote %s' % fout)

class ForceFieldError(Exception):
    pass

def minimize(pf, steps = 500, platform_name = 'CPU'):
    from simtk.openmm import app
    from simtk import openmm as mm
    from simtk import unit
    forcefield = app.ForceField('amber99sbildn.xml', 'amber99_obc.xml')
    topology = pf.topology
    try:
        system = forcefield.createSystem(topology, 
                                         nonbondedMethod=app.CutoffNonPeriodic,
                                         nonbondedCutoff=1.0*unit.nanometers,
                                         # Can't have hbond constraints with 0 mass fixed particles.
                                         constraints=app.HBonds,
                                         rigidWater=True)
    except ValueError as e:
        raise ForceFieldError('Missing atoms or parameterization needed by force field.\n' +
                              'All heavy atoms and hydrogens with standard names are required.\n' +
                              str(e))

    time_step = 2.0*unit.femtoseconds
    temperature = 0*unit.kelvin
    friction = 1.0/unit.picoseconds	# Coupling to heat bath
    constraint_tolerance = 0.001

    integrator = mm.LangevinIntegrator(temperature, friction, time_step)
    integrator.setConstraintTolerance(constraint_tolerance)
    platform = mm.Platform.getPlatformByName(platform_name)
    simulation = app.Simulation(topology, system, integrator, platform)
    c = simulation.context
    c.setPositions(pf.positions)
    c.setVelocitiesToTemperature(temperature)

    # Exceptions occur for bad structures.  Keep trying.
    step_block = 10
    for i in range(steps//step_block):
        try:
            simulation.minimizeEnergy(maxIterations = step_block)
        except Exception as e:
            if not 'Particle coordinate is nan' in str(e):
                raise
    state = c.getState(getPositions = True)
    pf.positions = state.getPositions()
    
def register_addmissingatoms_command():
    from chimerax.core.commands import CmdDesc, register, AtomicStructuresArg, BoolArg, IntArg
    desc = CmdDesc(
        required = [('structures', AtomicStructuresArg)],
        keyword = [('minimization_steps', IntArg),
                   ('keep_waters', BoolArg)],
        synopsis = 'Add missing heavy atoms and hydrogens to proteins using PDBFixer'
        )
    register('addmissingatoms', desc, addmissingatoms)
