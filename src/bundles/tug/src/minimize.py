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

# Mouse mode to click on residues an minimize using OpenMM that residue and nearby residues.
#

from chimerax.mouse_modes import MouseMode
class MinimizeMode(MouseMode):
    name = 'minimize'
    icon_file = 'minimize.png'

    def __init__(self, session):

        MouseMode.__init__(self, session)

        self._zone_range = 3		# Angstroms from chosen residue
        self._minimize_steps = 10	# Number of steps per coordinate update
        self._tugger = None
        self._minimize_handler = None
            
    def mouse_down(self, event):
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
                from .tugatoms import ForceFieldError
                try:
                    from .tugatoms import StructureTugger
                    self._tugger = st = StructureTugger(s)
                except ForceFieldError as e:
                    self.session.logger.warning(str(e))
                    return
            else:
                # Update simulation coordinates in case user moved atoms
                st._set_particle_positions()

            # Must set mobile atoms before creating OpenMM simulation
            self._set_mobile_atoms(a)
            st._make_simulation()
            
            self._minimize()
        
    def mouse_up(self, event = None):
        th = self._minimize_handler
        if th:
            self.session.triggers.remove_handler(th)
            self._minimize_handler = None

    def _set_mobile_atoms(self, atom):
        zatoms = _zone_atoms(atom.structure.atoms, atom.residue.atoms, self._zone_range)
        zra = zatoms.unique_residues.atoms
        self._tugger.mobile_atoms(zra)
        
    def _minimize(self, *_):
        if self._minimize_handler is None:
            self._minimize_handler = self.session.triggers.add_handler('new frame', self._minimize)
#        self._tugger._minimize(steps = self._minimize_steps)
        self._tugger._simulate(steps = self._minimize_steps)

    def vr_press(self, event):
        # Virtual reality hand controller button press.
        view = self.session.main_view
        pick = event.picked_object(view)
        self._pick_atom(pick)
        
    def vr_release(self, release):
        # Virtual reality hand controller button release.
        self.mouse_up()

def _zone_atoms(atoms, near_atoms, distance):
    axyz = atoms.scene_coords
    naxyz = near_atoms.scene_coords
    from chimerax.geometry import find_close_points
    i1,i2 = find_close_points(axyz, naxyz, distance)
    za = atoms[i1]
    return za
    
def register_mousemode(session):
    mm = session.ui.mouse_modes
    mm.add_mode(MinimizeMode(session))
