# vim: set expandtab ts=4 sw=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

# Mouse mode to drag atoms and update structure with OpenMM molecular dynamics.
#
# Structures need a complete set of heavy atoms and hydrogens.  OpenMM can add hydrogens but only
# if the heavy atoms are complete.  Most PDB models have partial side-chains.  Chimera 1 dock prep
# completes the partial side chains using the rotamers tools.  A simple case for adding all hydrogens
# is 1gcn -- testing shows hydrogens were added.  Other small NMR structures have all hydrogens such
# as 1mtx.  The openmm pdb format reader can handle 1mtx correctly apparently taking only the first model
# of the ensemble.  But the openmm mmcif reader does not work with 1mtx saying the atoms are right but
# wrong number of bonds for the N terminal threonine -- probably it combined all the atoms from the
# ensemble members.
#
# The current code below has openmm read the PDB or mmCIF file directly instead of creating the openmm
# topology from the structure that is open in ChimeraX.  It looks easy to create the openmm topology instead
# of rereading the file.
#
# I tried Chimera 1 dockprep on 1a0s.  OpenMM complained it did not have templates for calcium ions.
# After deleting 3 coordinated calcium ions it worked.  But within a minute it used 90 Gbytes and had
# to be force quit. Possibly the new frame callback kept calling simulate because the mouse up event
# was lost.  Doesn't make sense because it should have rendered a frame each calculation update but
# I only saw a few graphics updates.  Moving villin 35 residue example for a minute or two did not
# increase memory use much, mabye 100 Mbytes.
# Probably should run simulation in a separate thread to keep the gui responsive which would
# probably avoid losing the mouse event.
#
# Tried fixing positions of some atoms by setting particle mass to 0.  Did not appear to speed up
# simulation with villin with first half of particles with mass 0.  Had to remove bond length
# constraints to hydrogens which allows longer time steps also
# since error said constraints not allowed connecting to 0 mass particles.  Testing showed that it did
# avoid passing molecule through fixed atoms, so their forces are considered.
# Would need to make a fragment system to speed up calculation for larger systems.
#
# The OpenMM authors have another Python package PDBFixer that adds heavy atoms and hydrogens and makes
# use of OpenMM.  Might be worth trying that.  Also could apply it on the fly to prepare a 20 residue fragment
# for interactive MD.  It can build missing segments which could be nice for fitting in high res cryoEM.
#

openmm_forcefield_parameters = ['amber14-all.xml', 'amber14/tip3p.xml']
write_logs = False

from chimerax.mouse_modes import MouseMode
class TugAtomsMode(MouseMode):
    name = 'tug'
    icon_file = 'tug.png'

    def __init__(self, session):

        MouseMode.__init__(self, session)

        self._tugger = None
        self._tugging = False
        self._tug_handler = None
        self._tug_atom = None
        self._last_frame_number = None
        self._puller = None
        self._arrow_model = None

        self._log = Logger('tug.log' if write_logs else None)
            
    def mouse_down(self, event):
        self._log('In mouse_down')
        x,y = event.position()
        view = self.session.main_view
        pick = view.picked_object(x,y)
        self._pick_atom(pick)

    def _pick_atom(self, pick):
        if hasattr(pick, 'atom'):
            a = pick.atom
            st = self._tugger
            s = a.structure
            if st is None or not st.same_structure(s):
                try:
                    self._tugger = st = StructureTugger(a.structure)
                except ForceFieldError as e:
                    self.session.logger.warning(str(e))
                    return
            self._tug_atom = a
            from chimerax.atomic import Atoms
            st.tug_atoms(Atoms([a]))
            self._tugging = True

    def mouse_drag(self, event):
        self._log('In mouse_drag')
        x,y = event.position()
        self._puller = Puller2D(x,y)
        self._continue_tugging()
        
    def mouse_up(self, event = None):
        self._log('In mouse_up', close = True)
        self._tugging = False
        self._last_frame_number = None
        th = self._tug_handler
        if th:
            self.session.triggers.remove_handler(th)
            self._tug_handler = None
            a = self._arrow_model
            if a and not a.deleted:
                a.display = False
        
    def _tug(self):
        self._log('In _tug')
        if not self._tugging:
            return
        v = self.session.main_view
        if v.frame_number == self._last_frame_number:
            return	# Make sure we draw a frame before doing another MD calculation
        self._last_frame_number = v.frame_number

        a = self._tug_atom
        atom_xyz, target_xyz = self._puller.pull_to_point(a)

        self._tugger.tug_to_positions(target_xyz.reshape((1,3)))

        atom_xyz, target_xyz = self._puller.pull_to_point(a)
        self._draw_arrow(target_xyz, atom_xyz)

    def _continue_tugging(self, *_):
        self._log('In continue_tugging')
        if self._tug_handler is None:
            self._tug_handler = self.session.triggers.add_handler('new frame', self._continue_tugging)
        self._tug()

    def _draw_arrow(self, xyz1, xyz2, radius = 0.1):
        self._log('In draw_arrow')
        a = self._arrow_model
        if a is None or a.deleted:
            from chimerax.core.models import Model
            s = self.session
            self._arrow_model = a = Model('Tug arrow', s)
            from chimerax.surface import cone_geometry
            v,n,t = cone_geometry(points_up = False)
            a.set_geometry(v, n, t)
            a.color = (0,255,0,255)
            s.models.add([a])
        # Scale and rotate prototype cylinder.
        from chimerax.atomic import structure
        from numpy import array, float32
        p = structure._bond_cylinder_placements(array(xyz1).reshape((1,3)),
                                                array(xyz2).reshape((1,3)),
                                                array([radius],float32))
        a.position = p[0]
        a.display = True

    def vr_press(self, event):
        # Virtual reality hand controller button press.
        view = self.session.main_view
        pick = event.picked_object(view)
        self._pick_atom(pick)
        
    def vr_motion(self, event):
        # Virtual reality hand controller motion.
        self._puller = Puller3D(event.tip_position)
        self._continue_tugging()
        
    def vr_release(self, release):
        # Virtual reality hand controller button release.
        self.mouse_up()

class Puller2D:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def pull_to_point(self, atom):
        v = atom.structure.session.main_view
        x0,x1 = v.clip_plane_points(self.x, self.y)
        axyz = atom.scene_coord
        # Project atom onto view ray to get displacement.
        dir = x1 - x0
        from chimerax.geometry import inner_product
        pxyz = x0 + (inner_product(axyz - x0, dir)/inner_product(dir,dir)) * dir
        return axyz, pxyz

class Puller3D:
    def __init__(self, xyz):
        self.xyz = xyz
        
    def pull_to_point(self, atom):
        axyz = atom.scene_coord
        return axyz, self.xyz

class StructureTugger:
    '''
    Run a molecular dynamics simulation or minimization with OpenMM
    on a structure while pulling on some atoms.
    '''
    def __init__(self, structure, force_constant = 10000.0,
                 cutoff = 10.0, temperature = 100.0,
                 steps = 50, tolerance = 0.001):
        self._log = Logger('structuretugger.log' if write_logs else None)
        self.structure = structure
        self._minimized = False

        # OpenMM requires the atoms to be sorted by residue number
        self._structure_atoms = sa = structure.atoms
        satoms = list(sa)
        satoms.sort(key = lambda a: (a.residue.chain_id, a.residue.number))
        from chimerax.atomic import Atoms
        self.atoms = Atoms(satoms)

        initialize_openmm()
        
        # OpenMM objects
        self._topology = None
        self._system = None
        self._force = None	# CustomExternalForce pulling force
        self._platform = None
        self._simulation = None
        self._sim_forces = None # Current forces on atoms
        

        # OpenMM simulation parameters
        global openmm_forcefield_parameters
        self._forcefields = openmm_forcefield_parameters
        self._sim_steps = steps		# Simulation steps between mouse position updates
        self._force_constant = force_constant
        self._nonbonded_cutoff = cutoff
        from openmm import unit
        self._temperature = temperature * unit.kelvin
        self._integrator_tolerance = tolerance
        self._constraint_tolerance = tolerance
        self._friction = 1.0/unit.picoseconds	# Coupling to heat bath
        self._platform_name = 'CPU'
        #self._platform_name = 'OpenCL' # Works on Mac
        #self._platform_name = 'CUDA'	# This is 3x faster but requires env DYLD_LIBRARY_PATH=/usr/local/cuda/lib Chimera.app/Contents/MacOS/ChimeraX so paths to cuda libraries are found.
        self._max_allowable_force = 50000.0 # kJ/mol/nm
        
        
        # OpenMM particle data
        self._tugged_particle_numbers = None	# Integer indices of tugged atoms, numpy array
        self._particle_positions = None		# Positions for all atoms, numpy array, Angstroms
        self._particle_force_index = {}		# Maps particle number to force index for tugged atoms
        self._particle_masses = None		# Original particle masses

        self._create_openmm_system()

    def same_structure(self, structure):
        return (structure is self.structure and
                structure.atoms == self._structure_atoms)
    
    def tug_atoms(self, atoms):
        self._log('In tug_atoms')

        # OpenMM does not allow removing a force from a system.
        # So when the atom changes we either have to make a new system or add
        # the new atom to the existing force and set the force constant to zero
        # for the previous atom. Use the latter approach.
        new_p = self.atoms.indices(atoms)
        last_p = self._tugged_particle_numbers
        self._tugged_particle_numbers = new_p
        f = self._force
        pfi = self._particle_force_index
        k = self._force_constant
        x0 = y0 = z0 = 0

        # Reset force to 0 for last tugged particles that are no longer tugged.
        if last_p is not None:
            np = set(new_p)
            for lp in last_p:
                if lp not in np and lp in pfi:
                    f.setParticleParameters(pfi[lp], lp, (0,x0,y0,z0))

        # Set force parameters or add force term for this particle
        for p in new_p:
            if p in pfi:
                f.setParticleParameters(pfi[p], p, (k,x0,y0,z0))
            else:
                pfi[p] = f.addParticle(p, (k,x0,y0,z0))

        # If a particle is added to a force an existing simulation using
        # that force does not get updated. So we create a new simulation each
        # time a new atom is pulled. The integrator can only be associated with
        # one simulation so we also create a new integrator.
        self._make_simulation()

    def _make_simulation(self):
        import openmm as mm
        integrator = mm.VariableLangevinIntegrator(self._temperature, self._friction, self._integrator_tolerance)
        integrator.setConstraintTolerance(self._constraint_tolerance)

        # Make a new simulation.
        from openmm import app
        s = app.Simulation(self._topology, self._system, integrator, self._platform)
        self._simulation = s
        
    def _max_force (self):
        c = self._simulation.context
        from openmm.unit import kilojoule_per_mole, nanometer
        self._sim_forces = c.getState(getForces = True).getForces(asNumpy = True)/(kilojoule_per_mole/nanometer)
        forcesx = self._sim_forces[:,0]
        forcesy = self._sim_forces[:,1]
        forcesz = self._sim_forces[:,2]
        import numpy
        magnitudes =numpy.sqrt(forcesx*forcesx + forcesy*forcesy + forcesz*forcesz)
        return max(magnitudes)

    def tug_to_positions(self, xyz):
        '''xyz is in scene coordinates, one position for each tugged particle.'''
        self._log('In tug_to_positions')
        if len(xyz) != len(self._tugged_particle_numbers):
            raise ValueError('tug_to_positions: wrong number of tug points %d, expected %d'
                             % (len(xyz), len(self._tugged_particle_numbers)))
        self._set_tug_positions(xyz)
        if not self._minimized:
            self._minimize()
        self._simulate()

    def _set_tug_positions(self, xyz):
        '''xyz is in scene coordinates, one position for each tugged particle.'''
        txyz = self.structure.scene_position.inverse() * xyz        
        txyz *= 0.1	# convert position in Angstroms to nanometers
        f = self._force
        k = self._force_constant
        for (x0,y0,z0), p in zip(txyz, self._tugged_particle_numbers):
            fi = self._particle_force_index[p]
            f.setParticleParameters(fi, p, (k,x0,y0,z0))
        f.updateParametersInContext(self._simulation.context)

    def _simulate(self, steps = None):
# Minimization generates "Exception. Particle coordinate is nan" errors rather frequently, 1 in 10 minimizations.
#        simulation.minimizeEnergy(maxIterations = self._sim_steps)
# Get same error in LangevinIntegrator_step().
        simulation = self._simulation
        self._set_simulation_coordinates()
        sim_steps = self._sim_steps if steps is None else steps
        from time import time,sleep
        t0 = time()
        try:
            simulation.step(sim_steps)
            self._log('Did simulation step')
            max_force = self._max_force()
            if max_force > self._max_allowable_force:
                raise Exception('Maximum force exceeded')
            self._log('Maximum force: %.3g' % max_force)
        except Exception:
                max_force=self._max_force()
                self._log("FAIL!!!\n")
                self._set_simulation_coordinates()
                while (max_force > self._max_allowable_force):
                    self._log('Maximum force exceeded, %g > %g! Minimizing...'
                              % (max_force, self._max_allowable_force))
                    simulation.minimizeEnergy(maxIterations = sim_steps)
                    max_force = self._max_force()
        t1 = time()
        if write_logs:
            import sys
            sys.__stderr__.write('%d steps in %.2f seconds\n' % (sim_steps, t1-t0))
        self._update_atom_coords()

    def _minimize(self, steps = None):
        self._set_simulation_coordinates()
        min_steps = self._sim_steps if steps is None else steps
        self._simulation.minimizeEnergy(maxIterations = min_steps)
        self._update_atom_coords()
        self._minimized = True

    def mobile_atoms(self, atoms):
        '''
        Fix positions of some particles.  Must be called before creating OpenMM simulation otherwise
        it has no effect.

        This works by an OpenMM convention that zero mass particles do not move.
        But the OpenMM docs says contraints to zero mass particles don't work.
        This means bond length constraints cannot be used to allow longer integration
        time steps. For reasons I do not understand, OpenMM
        it will work.
        '''
        np = len(self.atoms)
        m = self._particle_masses
        system = self._system
        if m is None:
            self._particle_masses = m = [system.getParticleMass(i) for i in range(np)]
        mi = set(self.atoms.indices(atoms))
        freeze_mass = 0
        for i in range(np):
            mass = m[i] if i in mi else freeze_mass
            system.setParticleMass(i, mass)

    def _set_simulation_coordinates(self):
        c = self._simulation.context
        c.setPositions(0.1*self._particle_positions)	# Nanometers
        c.setVelocitiesToTemperature(self._temperature)
        
    def _set_particle_positions(self):
        self._particle_positions = self.atoms.coords
        
    def _simulation_atom_coordinates(self):
        c = self._simulation.context
        state = c.getState(getPositions = True)
        from openmm import unit
        pos = state.getPositions().value_in_unit(unit.angstrom)
        from numpy import array, float64
        xyz = array(pos, float64)
        return xyz

    def _update_atom_coords(self):
        xyz = self._simulation_atom_coordinates()
        self._particle_positions = xyz
        self.atoms.coords = xyz
        
    def _create_openmm_system(self):
        self._log('In create_openmm_system ')

        atoms = self.atoms
        self._particle_positions = atoms.coords
        bonds = self.structure.bonds
        self._topology = openmm_topology(atoms, bonds)

        from openmm import app
        forcefield = app.ForceField(*self._forcefields)

        self._add_ligands_to_forcefield(self.structure.residues, forcefield)
#        self._add_hydrogens(pdb, forcefield)

        self._system = system = self._create_system(forcefield)

#        path = '/Users/goddard/ucsf/amber/sustiva_tutorial/1FKO_sus.prmtop'
#        crd_path = path[:-7] + '.rst7'
#        path = '/Users/goddard/ucsf/amber/gfp_tutorial/gfp.parm7'
#        crd_path = '/Users/goddard/ucsf/amber/gfp_tutorial/min1.rst7'
#         self._system_from_prmtop(path, crd_path)
        
        import openmm as mm
        platform = mm.Platform.getPlatformByName(self._platform_name)
        self._platform = platform

        # Setup pulling force
        e = 'k*((x-x0)^2+(y-y0)^2+(z-z0)^2)'
        self._force = force = mm.CustomExternalForce(e)
        for name in ('k', 'x0', 'y0', 'z0'):
            force.addPerParticleParameter(name)
        system.addForce(force)

    def _add_ligands_to_forcefield(self, residues, forcefield):
        param_dir = self._ligand_parameters_directory()
        if param_dir is None:
            return     # TugLigands toolshed bundle not installed
        
        known_ligands = set(forcefield._templates.keys())
        known_ligands.add('HIS')	# Seems OpenMM handles this specially.
        from numpy import unique
        rnames = [rname for rname in unique(residues.names) if rname not in known_ligands]
        if len(rnames) == 0:
            return

        # Load GAFF atom types needed by Moriarty and Case ligand parameterizations.
        if not hasattr(self, '_gaff_types_added'):
            # Moriarty and Case parameterization uses GAFF atom types.
            from os import path
            gaff_types = path.join(param_dir, 'gaff2.xml')
            forcefield.loadFile(gaff_types)
            
        # Try getting ligand parameters from Moriarty and Case parameterization of all PDB ligands.
        rnames_not_found = []
        from os import path
        mc_ligand_zip = path.join(param_dir, 'moriarty_and_case_ligands.zip')
        from zipfile import ZipFile
        with ZipFile(mc_ligand_zip) as zf:
            for rname in rnames:
                try:
                    with zf.open(rname+'.xml') as xml_file:
                        forcefield.loadFile(xml_file)
                except KeyError:
                    rnames_not_found.append(rname)

        # Warn about ligands without parameters
        log = residues[0].structure.session.logger
        if rnames_not_found:
            log.warning('Could not find OpenMM parameters for %d residues %s in %s'
                        % (len(rnames_not_found), ', '.join(rnames_not_found), mc_ligand_zip))

        # Note ligands with parameters.
        rnames_found = [rname for rname in rnames if rname not in rnames_not_found]
        if rnames_found:
            log.info('Using OpenMM parameters from Moriarty and Case for %d residues %s'
                     % (len(rnames_found), ', '.join(rnames_found)))

    def _ligand_parameters_directory(self):
        try:
            from chimerax import tug_ligands
        except ImportError:
            return None
        return tug_ligands.parameters_directory

    def _system_from_prmtop(self, prmtop_path, impcrd_path):
        # load in Amber input files
        prmtop = app.AmberPrmtopFile(prmtop_path)
        inpcrd = app.AmberInpcrdFile(incrd_path)
        from numpy import array, float64
        from openmm import unit
        positions = 10*array(inpcrd.positions.value_in_unit(unit.nanometers), float64)  # Angstroms
        self._particle_positions = positions

        # Ordered atoms to match inpcrd order.
        # PDB can have atoms for a chain not contiguous, e.g. chain A hetatm at end of file.
        # But inpcrd has reordered so all chain atoms are contiguous.
        atom_pos = atoms.scene_coords
        from chimerax.geometry import find_closest_points
        i1,i2,near = find_closest_points(positions, atom_pos, 1.0)
        from numpy import empty, int32
        ai = empty((len(i1),), int32)
        ai[i1] = near
        self.atoms = oatoms = atoms[ai]
#        diff = oatoms.scene_coords - positions
#        print ('diff', diff.max(), diff.min())
#        p2a = {tuple(int(x) for x in a.scene_coord):a for a in atoms}
#        oatoms = [p2a[tuple(int(x) for x in p)] for p in positions]
#        print('\n'.join('%s %s' %(str(a1), str(a2)) for a1,a2 in zip(atoms, oatoms)))
#        print ('inpcrd', positions[:10])
#        print ('atoms', atom_pos[:10])
        
        # prepare system and integrator
        system = prmtop.createSystem(nonbondedMethod=app.CutoffNonPeriodic,
                                     nonbondedCutoff=self._nonbonded_cutoff*unit.angstrom,
                                     constraints=app.HBonds,
                                     rigidWater=True,
                                     ewaldErrorTolerance=0.0005
        )
        self._system = system

    def _create_system(self, forcefield):
        # We first try with ignoreExternalBonds = False.  This will fail any amino acid chain
        # ends do not have proper end-capping atoms, a common situation when there are missing
        # segments.  The addh command does not add termination atoms for missing segments.
        #
        # Then we try creating the system with ignoreExternalBonds = True.  This fails if
        # disulfides between cysteines are present with amber14 forcefield this giving a multiple
        # matching templates for CYS (CYM, CYX) error probably because the cysteine sulfur is
        # missing an attached atom.
        #
        # constraints = HBonds means the length of covalent bonds involving
        # hydrogen atoms are fixed to allow larger integration time steps.
        # Constraints are not supported to atoms with particle mass = 0 which
        # indicates a fixed atom position, and will generate errors.
        from openmm import app
        from openmm import unit
        try:
            system = forcefield.createSystem(self._topology, 
                                             nonbondedMethod=app.CutoffNonPeriodic,
                                             nonbondedCutoff=self._nonbonded_cutoff*unit.angstrom,
                                             constraints=app.HBonds,
                                             rigidWater=True,
                                             ignoreExternalBonds=False)
        except Exception as e1:
            system = None
            err1 = e1

        if system is not None:
            return system
            
        try:
            system = forcefield.createSystem(self._topology, 
                                             nonbondedMethod=app.CutoffNonPeriodic,
                                             nonbondedCutoff=self._nonbonded_cutoff*unit.angstrom,
                                             constraints=app.HBonds,
                                             rigidWater=True,
                                             ignoreExternalBonds=True)
        except Exception as e2:
            raise ForceFieldError('Missing atoms or parameterization needed by force field.\n' +
                                  'All heavy atoms and hydrogens with standard names are required.\n' +
                                  '\nError with ignoreExternalBonds=False was\n' + str(err1) +
                                  '\nError with ignoreExternalBonds=True was\n' + str(e2))
        return system
    

    def _add_hydrogens(self, openmm_pdb, forcefield):
        # Need hydrogens to run simulation and most structures don't have them.
        # Most PDBs have partial residues (e.g. 1a0m with ARG with only CB) and adding hydrogens
        # give error of missing template in that case.  Eric says Chimera 1 uses rotamers tool
        # to extend the residues to include all heavy atoms.
        # When openmm does have all heavy atoms (1gcn) it preserves order of those atoms
        # but hydrogens are inserted after the heavy atom they are attached to.  Would need to
        # take that into account to map between ChimeraX structure and openmm structure with hydrogens.
        from openmm.app.modeller import Modeller
        m = Modeller(openmm_pdb.topology, openmm_pdb.positions)
        m.addHydrogens(forcefield)
        top, pos = m.getTopology(), m.getPositions()
        print('Before adding hydrogens')
        dump_topology(openmm_pdb.topology)
        print('After adding hydrogens')
        dump_topology(top)

class ForceFieldError(Exception):
    pass

class Logger:
    def __init__(self, filename = None):
        self.filename = filename
        self._log_file = None
    def __call__(self, message, close = False):
        if self.filename is None:
            return	# No logging
        f = self._log_file
        if f is None:
            self._log_file = f = open(self.filename,'w')
            self._log_counter = 0
        f.write(message)
        f.write(' %d' % self._log_counter)
        f.write("\n")
        f.flush()
        self._log_counter += 1
        if close:
            f.close()
            self._log_file = None

def openmm_topology(atoms, bonds):
    '''Make OpenMM topology from ChimeraX atoms and bonds.'''
    a = atoms
    n = len(a)
    r = a.residues
    aname = a.names
    ename = a.element_names
    rname = r.names
    rnum = r.numbers
    cids = r.chain_ids
    from openmm.app import Topology, Element
    top = Topology()
    cmap = {}
    rmap = {}
    atoms = {}
    for i in range(n):
        cid = cids[i]
        if not cid in cmap:
            cmap[cid] = top.addChain()	# OpenMM chains have no name.
        rid = (rname[i], rnum[i], cid)
        if not rid in rmap:
            rmap[rid] = top.addResidue(rname[i], cmap[cid])
        element = Element.getBySymbol(ename[i])
        atoms[i] = top.addAtom(aname[i], element, rmap[rid])
    a1, a2 = bonds.atoms
    for i1, i2 in zip(a.indices(a1), a.indices(a2)):
        top.addBond(atoms[i1], atoms[i2])
    return top

_openmm_initialized = False
def initialize_openmm():
    # On linux need to set environment variable to find plugins.
    # Without this it gives an error saying there is no "CPU" platform.
    global _openmm_initialized
    if not _openmm_initialized:
        _openmm_initialized = True
        from sys import platform
        if platform == 'linux' or platform == 'darwin':
            from os import environ, path
            from chimerax import app_lib_dir
            environ['OPENMM_PLUGIN_DIR'] = path.join(app_lib_dir, 'plugins')
        
def dump_topology(t):
    for a in t.atoms():
        an, r, ai = a.name, a.residue, a.index
        rn, ri, c = r.name, r.index, r.chain
        ci = c.index
        print ('%d %s %s %d %d %s' % (ai, an, rn, ri, ci, c.id))

def register_mousemode(session):
    mm = session.ui.mouse_modes
    mm.add_mode(TugAtomsMode(session))
