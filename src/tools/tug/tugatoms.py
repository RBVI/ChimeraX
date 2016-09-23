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
        self._arrow_model = None
        
    def mouse_down(self, event):
        MouseMode.mouse_down(self, event)
        x,y = event.position()
        view = self.session.main_view
        pick = view.first_intercept(x,y)
        if hasattr(pick, 'atom'):
            a = pick.atom
            st = self._tugger
            if st is None or st.structure is not a.structure:
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
            a = self._arrow_model
            if a:
                a.display = False
        
    def _tug(self, x, y):
        if not self._tugging:
            return
        v = self.session.main_view
        if v.frame_number == self._last_frame_number:
            return	# Make sure we draw a frame before doing another MD calculation

        atom_xyz, offset = self._pull_direction(x, y)
        from time import time
        t0 = time()
        if self._tugger.tug_displacement(offset):
            self._last_frame_number = v.frame_number
        t1 = time()
        #print ('one pull time %.2f' % (t1-t0))
        atom_xyz, offset = self._pull_direction(x, y)
        self._draw_arrow(atom_xyz+offset, atom_xyz)

    def _pull_direction(self, x, y):
        v = self.session.main_view
        x0,x1 = v.clip_plane_points(x, y)
        axyz = self._tugger.atom.scene_coord
        # Project atom onto view ray to get displacement.
        dir = x1 - x0
        da = axyz - x0
        from chimerax.core.geometry import inner_product
        offset = da - (inner_product(da, dir)/inner_product(dir,dir)) * dir
        return axyz, -offset

    def _continue_tugging(self, *_):
        self._tug(*self._last_xy)

    def _draw_arrow(self, xyz1, xyz2, radius = 0.1):
        a = self._arrow_model
        if a is None:
            from chimerax.core.models import Model
            s = self.session
            self._arrow_model = a = Model('Tug arrow', s)
            from chimerax.core.surface import cone_geometry
            a.vertices, a.normals, a.triangles  = cone_geometry()
            a.color = (0,255,0,255)
            s.models.add([a])
        # Scale and rotate prototype cylinder.
        from chimerax.core.atomic import structure
        from numpy import array, float32
        p = structure._bond_cylinder_placements(xyz1.reshape((1,3)),
                                                xyz2.reshape((1,3)),
                                                array([radius],float32))
        a.position = p[0]
        a.display = True

class StructureTugger:
    def __init__(self, structure):
        self.structure = structure
        self.atoms = structure.atoms
        self.atom = None

        initialize_openmm()
        
        # OpenMM objects
        self._topology = None
        self._system = None
        self._force = None	# CustomExternalForce pulling force
        self._platform = None
        self._simulation = None

        # OpenMM simulation parameters
        self._sim_steps = 50		# Simulation steps between mouse position updates
        self._force_constant = 10000
        from simtk import unit
        #self._temperature = 300*unit.kelvin
        self._temperature = 0*unit.kelvin
        #self._constraint_tolerance = 0.00001
        self._time_step = 2.0*unit.femtoseconds
        self._constraint_tolerance = 0.001
        self._friction = 1.0/unit.picoseconds	# Coupling to heat bath
        self._platform_name = 'CPU'
        #self._platform_name = 'OpenCL' # Works on Mac
        #self._platform_name = 'CUDA'	# This is 3x faster but requires env DYLD_LIBRARY_PATH=/usr/local/cuda/lib Chimera.app/Contents/MacOS/ChimeraX so paths to cuda libraries are found.
        
        # OpenMM particle data
        self._particle_number = None		# Integer index of tugged atom
        self._particle_positions = None		# Numpy array, Angstroms
        self._particle_force_index = {}

        self._create_openmm_system()
        
    def tug_atom(self, atom):

        # OpenMM does not allow removing a force from a system.
        # So when the atom changes we either have to make a new system or add
        # the new atom to the existing force and set the force constant to zero
        # for the previous atom. Use the latter approach.
        self.atom = atom
        p = self.structure.atoms.index(atom)
        pp = self._particle_number
        f = self._force
        pfi = self._particle_force_index
        if pp is not None and pp != p and pp in pfi:
            f.setParticleParameters(pfi[pp], pp, (0,))	# Reset force to 0 for previous particle
        self._particle_number = p
        k = self._force_constant
        if p in pfi:
            f.setParticleParameters(pfi[p], p, (k,))
        else:
            pfi[p] = f.addParticle(p, (k,))

        # If a particle is added to a force an existing simulation using
        # that force does not get updated. So we create a new simulation each
        # time a new atoms is pulled. The integrator can only be associated with
        # one simulation so we also create a new integrator.
        from simtk import openmm as mm
        integrator = mm.LangevinIntegrator(self._temperature, self._friction, self._time_step)
        integrator.setConstraintTolerance(self._constraint_tolerance)

        # Make a new simulation.
        from simtk.openmm import app
        s = app.Simulation(self._topology, self._system, integrator, self._platform)
        self._simulation = s
        
    def tug_displacement(self, d):

        particle = self._particle_number
        pos = self._particle_positions
        pxyz = 0.1*pos[particle]	# Nanometers
        txyz = pxyz + 0.1*d		# displacement d is in Angstroms, convert to nanometers
        
        simulation = self._simulation
        c = simulation.context
        for p,v in zip(('x0','y0','z0'), txyz):
            c.setParameter(p,v)
        c.setPositions(0.1*pos)	# Nanometers
        c.setVelocitiesToTemperature(self._temperature)

# Minimization generates "Exception. Particle coordinate is nan" errors rather frequently, 1 in 10 minimizations.
#        simulation.minimizeEnergy(maxIterations = self._sim_steps)
# Get same error in LangevinIntegrator_step().
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
        #print ('%d steps in %.2f seconds' % (self._sim_steps, t1-t0))
        state = c.getState(getPositions = True)
        from simtk import unit
        pos = state.getPositions().value_in_unit(unit.angstrom)
        from numpy import array, float64
        self._particle_positions = xyz = array(pos, float64)
        self.atoms.coords = xyz
        return True
        
    def _create_openmm_system(self):
        from simtk.openmm import app
        from simtk import openmm as mm
        from simtk import unit

        self._topology, self._particle_positions = openmm_topology_and_coordinates(self.structure)
        
        forcefield = app.ForceField('amber99sbildn.xml', 'amber99_obc.xml')
#        self._add_hydrogens(pdb, forcefield)
        
        system = forcefield.createSystem(self._topology, 
                                         nonbondedMethod=app.CutoffNonPeriodic,
                                         nonbondedCutoff=1.0*unit.nanometers,
# Can't have hbond constraints with 0 mass fixed particles.
                                         constraints=app.HBonds,
                                         rigidWater=True)
        self._system = system

        # Fix positions of some particles
        # Test.
#        for i in range(len(self._particle_positions)//2):
#            system.setParticleMass(i, 0)

        platform = mm.Platform.getPlatformByName(self._platform_name)
        self._platform = platform

        # Setup pulling force
        e = 'k*((x-x0)^2+(y-y0)^2+(z-z0)^2)'
        self._force = force = mm.CustomExternalForce(e)
        force.addPerParticleParameter('k')
        for p in ('x0', 'y0', 'z0'):
            force.addGlobalParameter(p, 0.0)
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

def openmm_topology_and_coordinates(mol):
    '''Make OpenMM topology and positions from ChimeraX AtomicStructure.'''
    a = mol.atoms
    n = len(a)
    r = a.residues
    aname = a.names
    ename = a.element_names
    rname = r.names
    rnum = r.numbers
    cids = r.chain_ids
    from simtk.openmm.app import Topology, Element
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
    a1, a2 = mol.bonds.atoms
    for i1, i2 in zip(a1.indices(a), a2.indices(a)):
        top.addBond(atoms[i1], atoms[i2])
    from simtk.openmm import Vec3
    pos = a.coords
    return top, pos

_openmm_initialized = False
def initialize_openmm():
    # On linux need to set environment variable to find plugins.
    # Without this it gives an error saying there is no "CPU" platform.
    global _openmm_initialized
    if not _openmm_initialized:
        _openmm_initialized = True
        from sys import platform
        if platform == 'linux':
            from os import environ, path
            from chimerax import app_lib_dir
            environ['OPENMM_PLUGINS_DIR'] = path.join(app_lib_dir, 'plugins')
        
def dump_topology(t):
    for a in t.atoms():
        an, r, ai = a.name, a.residue, a.index
        rn, ri, c = r.name, r.index, r.chain
        ci = c.index
        print ('%d %s %s %d %d %s' % (ai, an, rn, ri, ci, c.id))

        
def register_mousemode(session):
    mm = session.ui.mouse_modes
    mm.add_mode(TugAtomsMode(session))
