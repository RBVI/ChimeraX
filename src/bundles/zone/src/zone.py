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

from os.path import join, dirname
from chimerax.ui import MouseMode
class AtomZoneMouseMode(MouseMode):
    name = 'zone'
    icon_file = join(dirname(__file__), 'zone.png')

    def __init__(self, session):
        MouseMode.__init__(self, session)
        self._zone_center_atom = None
        self._residue_distance = 5
        self._label_distance = 4
        self._surface_distance = 8
        self._coil_width_thickness = (0.2, 0.2)
        self._helix_width_thickness = (0.6, 0.2)
        self._sheet_width_thickness = (0.6, 0.2)
        self._ribbon_transparency = 100			# 0 = completely transparent, 255 = opaque

    def _show_zone(self, atom = None, center = False, label=True, ribbon=True, ribbon_hiding = True):
        '''Show nearby atoms, labels, and surfaces.'''

        if atom is None:
            atom = self._zone_center_atom
        else:
            self._zone_center_atom = atom
            
        ratoms = atom.residue.atoms
        struct = atom.structure
        all_atoms = struct.atoms

        natoms = self._show_near_atoms(ratoms, all_atoms, show_ligands=ribbon)
        if label:
            self._show_labels(ratoms, all_atoms)
        if ribbon:
            self._show_ribbons(struct, natoms.unique_residues, ribbon_hiding)
        self._show_volume_zone(ratoms)

        # Set center of rotation
        v = self.session.main_view
        v.center_of_rotation = atom.scene_coord

        if center:
            self._center_camera(atom.scene_coord)

    def _unzone(self):
        atom = self._zone_center_atom
        if atom is None:
            return
        struct = atom.structure
        struct.atoms.displays = True
        from chimerax.map.filter.vopcommand import volume_unzone
        volume_unzone(self.session, self._shown_volumes())
        
    def _show_near_atoms(self, ratoms, all_atoms, show_ligands):
        natoms = self._nearby_atoms(ratoms, all_atoms, self._residue_distance)
        cats = all_atoms.structure_categories
        if show_ligands:
            all_atoms[cats == 'main'].displays = False	   # Keep ligands and ions shown
            all_atoms[cats == 'solvent'].displays = False
        else:
            all_atoms.displays = False
        natoms.displays = True
        return natoms
    
    def _nearby_atoms(self, atoms, other_atoms, distance):
        from chimerax.std_commands.zonesel import zone_items
        za, zs = zone_items(atoms, [], distance, other_atoms, [], extend = True, residues = True)
        return za

    def _show_labels(self, atoms, all_atoms):
        # Show residue labels for nearby atoms.
        latoms = self._nearby_atoms(atoms, all_atoms, self._label_distance)
        from chimerax.core.objects import Objects
        aobj = Objects(atoms = latoms)
        from chimerax.label.label3d import label, label_delete
        ses = self.session
        label_delete(ses)
        from chimerax.core.colors import BuiltinColors
        label(ses, aobj, 'residues', size = 64, height = 0.7, orient = 45,
              color = BuiltinColors['yellow'], background = BuiltinColors['black'])

    def _show_ribbons(self, struct, hide_residues, ribbon_hiding):
        # Show ribbons thinner and transparent
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
        if ribbon_hiding:
            res.ribbon_displays = True
            hide_residues.ribbon_displays = False
        else:
            # Make ribbon where atoms are shown completely transparent
            self._set_ribbon_transparency(res, self._ribbon_transparency)
            self._set_ribbon_transparency(hide_residues, 0)

    def _set_ribbon_transparency(self, residues, alpha):
        rcolors = residues.ribbon_colors
        rcolors[:,3] = alpha
        residues.ribbon_colors = rcolors

    def _show_volume_zone(self, atoms):
        from chimerax.map.filter.zone import zone_operation
        for v in self._shown_volumes():
            zone_operation(v, atoms, self._surface_distance, minimal_bounds = True, new_map = False)

    def _shown_volumes(self):
        # Show surface zone
        from chimerax.map import Volume
        ses = self.session
        volumes = [v for v in ses.models.list(type = Volume) if v.visible]
        return volumes

    def _center_camera(self, scene_point):
        # Center view on atom by moving camera perpendicular to sight direction.
        # This pretty disorienting since everything in view jumps.
        v = self.session.main_view
        cpos = v.camera.position
        (ax,ay,az) = cpos.inverse() * scene_point
        offset = cpos*(ax,ay,0) - cpos.origin()
        from chimerax.core.geometry import translation
        v.move(translation(-offset))
    
    def mouse_down(self, event):
        MouseMode.mouse_down(self, event)
        atom = self._mouse_pick(event)
        if atom is not None:
            self._show_zone(atom)
    
    def mouse_drag(self, event):
        # TODO: Change radius by dragging?  Could be slow.
        pass

    def mouse_up(self, event):
        MouseMode.mouse_up(self, event)
    
    def wheel(self, event):
        # TODL: Change radius of zone?
        pass
    
    def _mouse_pick(self, event):
        x,y = event.position()
        from chimerax.ui.mousemodes import picked_object
        pick = picked_object(x, y, self.session.main_view)
        return self._picked_atom(pick)
    
    def _picked_atom(self, pick):
        from chimerax.atomic import PickedAtom, PickedResidue
        if isinstance(pick, PickedAtom):
            a = pick.atom
        elif isinstance(pick, PickedResidue):
            a = pick.residue.principal_atom
        else:
            a = None
        return a

    def laser_click(self, xyz1, xyz2):
        from chimerax.ui.mousemodes import picked_object_on_segment
        pick = picked_object_on_segment(xyz1, xyz2, self.view)
        atom = self._picked_atom(pick) 
        if atom:
            self._show_zone(atom, center=False, label=True, ribbon=False)
        else:
            self._unzone()

    def drag_3d(self, position, move, delta_z):
        if move is None:
            # released button
            pass
        else:
            if abs(delta_z) < .1:
                return "accumulate drag"
            a = self._zone_center_atom
            if a:
                from math import exp
                scale = exp(delta_z / .3)
                self._residue_distance *= scale
                self._label_distance *= scale
                self._surface_distance *= scale
                self._show_zone(center=False, label=False, ribbon=False)
