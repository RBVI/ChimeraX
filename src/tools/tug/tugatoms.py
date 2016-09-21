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
# simulation with villin with first half of particles with mass 0.  Had to remove hbond constraints too
# since error said constraints not allowed connecting to 0 mass particles.  Testing showed that it did
# avoid passing molecule through fixed atoms, so their forces are considered.
# Would need to make a fragment system to speed up calculation for larger systems.
#
# The OpenMM authors have another Python package PDBFixer that adds heavy atoms and hydrogens and makes
# use of OpenMM.  Might be worth trying that.  Also could apply it on the fly to prepare a 20 residue fragment
# for interactive MD.  It can build missing segments which could be nice for fitting in high res cryoEM.
#
from chimerax.core.ui import MouseMode
class TugAtomsMode(MouseMode):
    name = 'tug'
    icon_file = 'tug.png'

    def __init__(self, session):

        MouseMode.__init__(self, session)

        self._tugger = None
        self._tugging = False
        self._tug_handler = None
        self._last_frame_number = None
        self._last_xy = None
        
    def mouse_down(self, event):
        MouseMode.mouse_down(self, event)
        x,y = event.position()
        view = self.session.main_view
        pick = view.first_intercept(x,y)
        if hasattr(pick, 'atom'):
            a = pick.atom
            st = self._tugger
            if st is None:
                self._tugger = st = StructureTugger(a.structure)
            st.tug_atom(a)
            self._tugging = True

    def mouse_drag(self, event):
        self._last_xy = x,y = event.position()
        self._tug(x, y)
        if self._tug_handler is None:
            self._tug_handler = self.session.triggers.add_handler('new frame', self._continue_tugging)
        
    def mouse_up(self, event):
        MouseMode.mouse_up(self, event)
        self._tugging = False
        self._last_frame_number = None
        th = self._tug_handler
        if th:
            self.session.triggers.remove_handler(th)
            self._tug_handler = None
        
    def _tug(self, x, y):
        if not self._tugging:
            return
        st = self._tugger
        v = self.session.main_view
        if v.frame_number == self._last_frame_number:
            return	# Make sure we draw a frame before doing another MD calculation
        x0,x1 = v.clip_plane_points(x, y)
        axyz = st.atom.scene_coord
        # Project atom onto view ray to get displacement.
        dir = x1 - x0
        da = axyz - x0
        from chimerax.core.geometry import inner_product
        offset = da - (inner_product(da, dir)/inner_product(dir,dir)) * dir
        from time import time
        t0 = time()
        if st.tug_displacement(-offset):
            self._last_frame_number = v.frame_number
        t1 = time()
#        print ('one pull time %.2f' % (t1-t0))

    def _continue_tugging(self, *_):
        self._tug(*self._last_xy)

class StructureTugger:
    def __init__(self, structure):
        self.structure = structure
        self.atoms = structure.atoms
        self.atom = None

        self._particle_number = None
        self._particle_positions = None
        self._force = None
        self._force_terms = {}
        self._sim_steps = 50
        self._force_constant = 200
        self._openmm_platform = None
        self._openmm_system = None
        self._openmm_pdb = None

        self._create_openmm_system()
        
    def tug_atom(self, atom):
        self.atom = atom
        p = self.structure.atoms.index(atom)
        pp = self._particle_number
        f = self._force
        ft = self._force_terms
        if pp is not None and pp != p and pp in ft:
            f.setParticleParameters(ft[pp], pp, (0,0,0,0))	# Reset force to 0 for previous particle
        self._particle_number = p
        if not p in ft:
            ft[p] = f.addParticle(p, (0,0,0,0))
        
    def tug_displacement(self, d):

        from simtk.openmm import app
        from simtk import openmm as mm
        from simtk import unit
        
        particle = self._particle_number
        pos = self._particle_positions
        px,py,pz = pos[particle].value_in_unit(unit.nanometer)
        px += d[0]
        py += d[1]
        pz += d[2]
        self._force.setParticleParameters(self._force_terms[particle], particle, (px,py,pz,self._force_constant))
    
        a = self.atom
        r = a.residue
#        print('Moving particle %d, atom %s, res %s resnum %d in direction %.1f %.1f %.1f toward position %.1f %.1f %.1f' %
#              (particle, a.name, r.name, r.number, d[0], d[1], d[2], px, py, pz))

        # Require new integrator because previous simulation "owns" the integrator.
#        integrator = mm.LangevinIntegrator(300*unit.kelvin, 1.0/unit.picoseconds, 2.0*unit.femtoseconds)
#        integrator = mm.LangevinIntegrator(10*unit.kelvin, 1.0/unit.picoseconds, 2.0*unit.femtoseconds)
        integrator = mm.LangevinIntegrator(0*unit.kelvin, 1.0/unit.picoseconds, 2.0*unit.femtoseconds)
#        integrator.setConstraintTolerance(0.00001)
        integrator.setConstraintTolerance(0.001)

        # Make a new simulation.
        # Apparently the force parameters are compiled by the simulation and cannot be changed
        # after the simulation is made.
        simulation = app.Simulation(self._openmm_pdb.topology, self._openmm_system, integrator, self._openmm_platform)
        simulation.context.setPositions(pos)
#        simulation.context.setVelocitiesToTemperature(300*unit.kelvin)
        simulation.context.setVelocitiesToTemperature(0*unit.kelvin)

# Minimization tends "Particle coordinate is nan" errors rather frequently, 1 in 10 minimizations.
#        simulation.minimizeEnergy(maxIterations = self._sim_steps)
# Also get "Exception: Particle coordinate is nan" in LangevinIntegrator_step().
        from time import time
        t0 = time()
        try:
            simulation.step(self._sim_steps)
        except Exception as e:
            if 'Particle coordinate is nan' in str(e):
                return False
            else:
                raise
        t1 = time()
#        print ('%d steps in %.2f seconds' % (self._sim_steps, t1-t0))
        state = simulation.context.getState(getPositions = True)
        self._particle_positions = pos = state.getPositions()
        from numpy import array, float64
        xyz = array(pos.value_in_unit(unit.angstrom), float64)
        self.atoms.coords = xyz
        return True
        
    def _create_openmm_system(self):
        from simtk.openmm import app
        from simtk import openmm as mm
        from simtk import unit

        path = self.structure.filename
        if path.endswith('.pdb'):
            pdb = app.PDBFile(path)
        elif path.endswith('.cif'):
            pdb = app.PDBxFile(path)
        else:
            raise ValueError('Atom motion requires PDB or mmCIF format file, got %s' % pdb_path)
        self._openmm_pdb = pdb
        self._particle_positions = self._openmm_pdb.positions
        
        forcefield = app.ForceField('amber99sbildn.xml', 'amber99_obc.xml')
#        self._add_hydrogens(pdb, forcefield)
        
        system = forcefield.createSystem(pdb.topology, 
                                         nonbondedMethod=app.CutoffNonPeriodic,
                                         nonbondedCutoff=1.0*unit.nanometers,
# Can't have hbond constraints with 0 mass fixed particles.
                                         constraints=app.HBonds,
                                         rigidWater=True)
        self._openmm_system = system
        # Fix positions of some particles
        # Test.
#        for i in range(len(self._particle_positions)//2):
#            system.setParticleMass(i, 0)

        platform = mm.Platform.getPlatformByName('CPU')
#        platform = mm.Platform.getPlatformByName('CUDA')
        self._openmm_platform = platform

        # Setup pulling force
        k = self._force_constant
        e = 'k*((x-x0)^2+(y-y0)^2+(z-z0)^2)'
        self._force = force = mm.CustomExternalForce(e)
        param_indices = [force.addPerParticleParameter(param) for param in ('x0', 'y0', 'z0', 'k')]
        system.addForce(force)

    def _add_hydrogens(self, openmm_pdb, forcefield):
        # Need hydrogens to run simulation and most structures don't have them.
        # Most PDBs have partial residues (e.g. 1a0m with ARG with only CB) and adding hydrogens
        # give error of missing template in that case.  Eric says Chimera 1 uses rotamers tool
        # to extend the residues to include all heavy atoms.
        # When openmm does have all heavy atoms (1gcn) it preserves order of those atoms
        # but hydrogens are inserted after the heavy atom they are attached to.  Would need to
        # take that into account to map between ChimeraX structure and openmm structure with hydrogens.
        from simtk.openmm.app.modeller import Modeller
        m = Modeller(openmm_pdb.topology, openmm_pdb.positions)
        m.addHydrogens(forcefield)
        top, pos = m.getTopology(), m.getPositions()
        print('Before adding hydrogens')
        dump_topology(openmm_pdb.topology)
        print('After adding hydrogens')
        dump_topology(top)

def dump_topology(t):
    for a in t.atoms():
        an, r, ai = a.name, a.residue, a.index
        rn, ri, c = r.name, r.index, r.chain
        ci = c.index
        print ('%d %s %s %d %d %s' % (ai, an, rn, ri, ci, c.id))

        
def register_mousemode(session):
    mm = session.ui.mouse_modes
    mm.add_mode(TugAtomsMode(session))
