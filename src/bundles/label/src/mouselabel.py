from chimerax.ui import MouseMode

class LabelMouseMode(MouseMode):
    '''Click an atom,ribbon,pseudobond or bond to label or unlabel it with default label.'''
    name = 'label'
    icon_file = 'label.png'

    def mouse_down(self, event):
        MouseMode.mouse_down(self, event)
        x,y = event.position()
        from chimerax.ui.mousemodes import picked_object
        pick = picked_object(x, y, self.session.main_view)
        self._label_pick(pick)

    def _label_pick(self, pick, color = None, background = None, height = None, orient = None):
        if pick is None:
            return

        from chimerax import atomic
        if isinstance(pick, atomic.PickedAtom):
            atoms = pick.atom.residue.atoms
        elif isinstance(pick, atomic.PickedResidue):
            atoms = pick.residue.atoms
        elif isinstance(pick, atomic.PickedBond):
            atoms = atomic.Atoms(pick.bond.atoms)  # Convert tuple to Atoms
        elif isinstance(pick, atomic.PickedPseudobond):
            atoms = atomic.Atoms(pick.pbond.atoms)  # Convert tuple to Atoms
        else:
            return

        from chimerax.core.objects import Objects
        objects = Objects(atoms = atoms)
        object_type = 'residues'
        
        ses = self.session
        from chimerax.label.label3d import label, label_delete
        if label_delete(ses, objects, object_type) == 0:
            label(ses, objects, object_type, color=color, background=background,
                  height=height, orient=orient)

    def laser_click(self, xyz1, xyz2):
        from chimerax.ui.mousemodes import picked_object_on_segment
        pick = picked_object_on_segment(xyz1, xyz2, self.view)
        if pick:
            from chimerax.core.colors import BuiltinColors
            self._label_pick(pick,
                             color = BuiltinColors['yellow'],
                             background = BuiltinColors['darkslategray'],
                             height = 1,
                             orient = 45)
            # Use opaque background to speed up rendering and improve appearance in VR.
            # Use fixed height in scene units since that is more natural in VR.
            # Reorient only on 45 degree view changes, less distracting in VR.
