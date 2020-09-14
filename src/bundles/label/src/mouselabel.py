from chimerax.mouse_modes import MouseMode

class LabelMouseMode(MouseMode):
    '''Click an atom,ribbon,pseudobond or bond to label or unlabel it with default label.'''
    name = 'label'
    icon_file = 'label.png'

    def mouse_down(self, event):
        MouseMode.mouse_down(self, event)
        x,y = event.position()
        pick = self.session.main_view.picked_object(x, y)
        self._label_pick(pick)

    def _label_pick(self, pick, color = None, background = None,
                    size = None, height = None):
        if pick is None:
            from .label3d import label_delete
            label_delete(self.session, object_type = 'residues')
            from chimerax.core.commands import log_equivalent_command
            log_equivalent_command(self.session, 'label delete residues')
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
        from .label3d import label, label_delete
        if label_delete(ses, objects, object_type) == 0:
            label(ses, objects, object_type, color=color, bg_color=background,
                  size=size, height=height)
            from chimerax.core.commands import log_equivalent_command, residues_specifier, options_text
            rspec = residues_specifier(objects)
            opts = options_text((('color',color), ('bg_color',background), 
                                 ('size',size), ('height',height)))
            log_equivalent_command(ses, 'label %s %s' % (rspec, opts))

    def vr_press(self, event):
        # Virtual reality hand controller button press.
        pick = event.picked_object(self.view)
        from chimerax.core.colors import BuiltinColors
        self._label_pick(pick,
                         color = BuiltinColors['yellow'],
                         background = BuiltinColors['dimgray'],
                         size = 64,		# pixels
                         height = 0.7)	# Angstroms
        # Use opaque background to speed up rendering and improve appearance in VR.
        # Use fixed height in scene units since that is more natural in VR.
          
def register_mousemode(session):
    mm = session.ui.mouse_modes
    mm.add_mode(LabelMouseMode(session))
