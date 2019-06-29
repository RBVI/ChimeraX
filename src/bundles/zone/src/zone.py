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

from chimerax.mouse_modes import MouseMode
class AtomZoneMouseMode(MouseMode):
    name = 'zone'
    icon_file = 'zone.png'

    def __init__(self, session):
        MouseMode.__init__(self, session)
        self._zone_center_residue = None
        self._residue_distance = 5
        self._label_distance = 4
        self._label_size = 64		# Pixels
        self._label_height = 0.7	# Scene units (Angstroms)
        from chimerax.core.colors import BuiltinColors
        self._label_color = BuiltinColors['yellow']
        self._label_background = BuiltinColors['black']
        self._surface_distance = 8
        self._coil_width_thickness = (0.2, 0.2)
        self._helix_width_thickness = (0.6, 0.2)
        self._sheet_width_thickness = (0.6, 0.2)
        self._ribbon_transparency = 100		# 0 = completely transparent, 255 = opaque
        self._labeled_residues = None
        self._scale_accum = 1
        self._scale_step = 1.3			# Minimum scaling step factor
        self._original_atom_display = None	# Boolean mask of which atoms originally shown
        self._original_residue_display = None	# Boolean mask of which residues originally shown
        
    def _show_zone(self, residue, label=True, ribbon=True, log_command = True):
        '''Show nearby residues, labels, and surfaces.'''

        if self._zone_center_residue is None:
            # Remember original display state.
            s = residue.structure
            self._original_atom_display = s.atoms.displays
            self._original_residue_display = s.residues.ribbon_displays

        self._zone_center_residue = residue
        
        ratoms = residue.atoms
        nres = self._show_near_residues(residue, show_ligands=ribbon)

        if label:
            self._show_labels()

        if ribbon:
            self._show_ribbons(nres)

        self._show_volume_zone(ratoms)

        # Set center of rotation
        v = self.session.main_view
        c = residue.structure.scene_position * residue.center
        v.center_of_rotation = c

        if log_command:
            rspec = residue.string(style='command line')
            options = ' '.join(('' if ribbon else 'ribbon False',
                                '' if label else 'label False'))
            if log_command == 'include distances':
                options = ' '.join((options,
                                    'residueDistance %.3g' % self._residue_distance,
                                    'labelDistance %.3g' % self._label_distance,
                                    'surfaceDistance %.3g' % self._surface_distance))
            from chimerax.core.commands import log_equivalent_command
            log_equivalent_command(self.session, 'zone %s %s' % (rspec, options))

    def _unlabel(self, log_command = True):
        res = self._labeled_residues
        if res is None:
            return False
        from chimerax.label.label3d import label_delete
        label_delete(self.session, res, 'residues')
        self._labeled_residues = None
        if log_command:
            from chimerax.core.commands import log_equivalent_command
            log_equivalent_command(self.session, 'zone label False')
        return True

    def _unzone(self, log_command = True):
        r = self._zone_center_residue
        if r is None:
            return

        self._restore_original_display()
        self._unlabel(log_command = False)
        from chimerax.map.filter.vopcommand import volume_unzone
        volume_unzone(self.session, self._shown_volumes())
        if log_command:
            from chimerax.core.commands import log_equivalent_command
            log_equivalent_command(self.session, 'zone clear')

        self._zone_center_residue = None

    def _restore_original_display(self):
        r = self._zone_center_residue
        if r is None:
            return
        s = r.structure

        ad = self._original_atom_display
        if ad is not None and len(ad) == s.num_atoms:
            s.atoms.displays = ad
        self._original_atom_display = None

        rd = self._original_residue_display
        if rd is not None and len(rd) == s.num_residues:
            s.residues.ribbon_displays = rd
        self._original_residue_display = None
        
        
    def _show_near_residues(self, residue, show_ligands):
        nres = self._nearby_residues(residue, self._residue_distance)
        all_atoms = residue.structure.atoms
        cats = all_atoms.structure_categories
        if show_ligands:
            all_atoms[cats == 'main'].displays = False	   # Keep ligands and ions shown
            all_atoms[cats == 'solvent'].displays = False
        else:
            all_atoms.displays = False
        nres.atoms.displays = True
        return nres
    
    def _nearby_residues(self, residue, distance):
        from chimerax.std_commands.zonesel import zone_items
        atoms = residue.atoms
        other_atoms = residue.structure.atoms
        za, zs = zone_items(atoms, [], distance, other_atoms, [],
                            extend = True)
        nres = za.unique_residues
        return nres

    def _show_labels(self):
        # Show residue labels for nearby residues
        r = self._zone_center_residue
        lres = self._nearby_residues(r, self._label_distance)
        from chimerax.core.objects import Objects
        aobj = Objects(atoms = lres.atoms)
        from chimerax.label.label3d import label, label_delete
        ses = self.session
        label_delete(ses)
        label(ses, aobj, 'residues', size = self._label_size, height = self._label_height,
              color = self._label_color, background = self._label_background)
        self._labeled_residues = aobj

    def _show_ribbons(self, hide_residues):
        # Show ribbons thinner and transparent
        for struct in hide_residues.unique_structures:
            res = struct.residues
            if not hasattr(struct, '_zone_ribbon_setup'):
                struct._zone_ribbon_setup = True
                res.ribbon_displays = True
                res.ribbon_hide_backbones = False
                rm = struct.ribbon_xs_mgr
                cw,ch = self._coil_width_thickness
                rm.set_coil_scale(cw, ch)
                hw,hh = self._helix_width_thickness
                rm.set_helix_scale(hw, hh)
                sw,sh = self._sheet_width_thickness
                rm.set_sheet_scale(sw, sh)

            res.ribbon_displays = True
            hide_residues.ribbon_displays = False

    def _set_ribbon_transparency(self, residues, alpha):
        rcolors = residues.ribbon_colors
        rcolors[:,3] = alpha
        residues.ribbon_colors = rcolors

    def _show_volume_zone(self, atoms):
        from chimerax.map.filter.zone import zone_operation
        for v in self._shown_volumes():
            zone_operation(v, atoms, self._surface_distance,
                           minimal_bounds = True, new_map = False)

    def _shown_volumes(self):
        # Show surface zone
        from chimerax.map import Volume
        ses = self.session
        volumes = [v for v in ses.models.list(type = Volume) if v.visible]
        return volumes

    def _scale_range(self, scale, ribbon=True):
        r = self._zone_center_residue
        if r is None:
            return

        # Accumulate scaling until a large enough scaling is requested
        s = self._scale_accum * scale
        step = self._scale_step
        if s < step and s > 1/step:
            self._scale_accum = s
            return
        self._scale_accum = 1

        self._residue_distance *= s
        self._label_distance *= s
        self._surface_distance *= s

        self._show_zone(r, ribbon=ribbon, log_command = 'include distances')
    
    def mouse_down(self, event):
        MouseMode.mouse_down(self, event)
        res = self._mouse_pick(event)
        if res:
            self._show_zone(res)
        elif not self._unlabel():
            self._unzone()
    
    def mouse_drag(self, event):
        dx, dy = self.mouse_motion(event)
        from math import exp
        scale = exp(-0.003*dy)
        self._scale_range(scale)

    def mouse_up(self, event):
        MouseMode.mouse_up(self, event)
    
    def wheel(self, event):
        d = event.wheel_value()
        from math import exp
        scale = exp(0.3*d)
        self._scale_range(scale)
    
    def _mouse_pick(self, event):
        x,y = event.position()
        from chimerax.mouse_modes import picked_object
        pick = picked_object(x, y, self.session.main_view)
        return self._picked_residue(pick)
    
    def _picked_residue(self, pick):
        from chimerax.atomic import PickedAtom, PickedResidue, Atoms
        if isinstance(pick, PickedAtom):
            r = pick.atom.residue
        elif isinstance(pick, PickedResidue):
            r = pick.residue
        else:
            r = None
        return r

    def vr_press(self, xyz1, xyz2):
        # Virtual reality hand controller button press.
        from chimerax.mouse_modes import picked_object_on_segment
        pick = picked_object_on_segment(xyz1, xyz2, self.view)
        res = self._picked_residue(pick) 
        if res:
            self._show_zone(res, ribbon=False)
        elif not self._unlabel():
            self._unzone()

    def vr_motion(self, position, move, delta_z):
        # Virtual reality hand controller motion.
        from math import exp
        scale = exp(delta_z / .3)
        self._scale_range(scale, ribbon=False)

def zone(session, atoms = None, residue_distance = None,
         label = None, label_distance = None,
         surface_distance = None, ribbon = True):
    zm = session._atom_zone_mouse_mode
    if residue_distance is not None:
        zm._residue_distance = residue_distance
    if label_distance is not None:
        zm._label_distance = label_distance
    if surface_distance is not None:
        zm._surface_distance = surface_distance
    if atoms is None and label is not None:
        if label:
            zm._show_labels()
        else:
            zm._unlabel(log_command = False)
    if atoms is not None:
        res = atoms.unique_residues
        if len(res) != 1:
            from chimerax.core.errors import UserError
            raise UserError('Must specify a single residue, %d specified' % len(res))
        lbl = True if label is None else label
        zm._show_zone(res[0], label=lbl, ribbon=ribbon, log_command = False)

def zone_clear(session):
    zm = session._atom_zone_mouse_mode
    zm._unzone(log_command = False)
         
def register_zone_command(logger):
    from chimerax.core.commands import CmdDesc, register, BoolArg, FloatArg
    from chimerax.atomic import AtomsArg
    desc = CmdDesc(
        optional = [('atoms', AtomsArg)],
        keyword = [('label', BoolArg),
                   ('label_distance', FloatArg),
                   ('surface_distance', FloatArg),
                   ('residue_distance', FloatArg),
                   ('ribbon', BoolArg)],
        synopsis = 'Show atom and map zone'
    )
    register('zone', desc, zone, logger=logger)

    desc = CmdDesc(synopsis = 'Show all atoms and full map')
    register('zone clear', desc, zone_clear, logger=logger)
