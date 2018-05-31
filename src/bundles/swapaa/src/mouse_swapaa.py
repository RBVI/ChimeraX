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

from chimerax.ui import MouseMode
class SwapAAMouseMode(MouseMode):
    name = 'swapaa'
    icon_file = 'swapaa.png'

    def __init__(self, session):
        MouseMode.__init__(self, session)
        self._residue = None
        self._align_atom_names = ['N', 'C', 'CA']
        self._keep_atom_names = ['N', 'C', 'CA', 'O', 'H']
        self._step_pixels = 20
        self._step_meters = 0.05
        self._last_y = None
        self._template_residues = []
        self._template_residue_names = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS',
                                        'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
                                        'LEU', 'LYS', 'MET', 'PHE', 'PRO',
                                        'SER', 'THR', 'TRP', 'TYR', 'VAL']
        self._template_pdb_id = '5D8V'		# Source of atom coordinates for each residue.
        self._label_atom_name = 'CA'		# Which atom to show residue label on.
        
    def enable(self):
        self._load_templates()

    def _load_templates(self):
        tres = self._template_residues
        if tres:
            return
        from chimerax.core.atomic.mmcif import fetch_mmcif
        models, status = fetch_mmcif(self.session, self._template_pdb_id, log_info = False)
        m = models[0]
        found = {}
        tnames = self._template_residue_names
        # TODO: Exclude residues with terminal groups.
        for r in m.residues[1:-1]:
            if r.name not in found and r.name in tnames:
                found[r.name] = r
        tres.extend(found[name] for name in tnames if name in found)
        
    def mouse_down(self, event):
        MouseMode.mouse_down(self, event)
        r = self._picked_residue(event)
        if r and self._has_alignment_atoms(r):
            self._residue = r
            self._last_y = event.position()[1]
    
    def mouse_drag(self, event):
        r = self._residue
        if r:
            x,y = event.position()
            dy = y - self._last_y
            if abs(dy) >= self._step_pixels:
                rstep = dy / self._step_pixels
                if self._swap_residue_step(r, rstep):
                    self._last_y = y

    def mouse_up(self, event):
        self._unlabel()
        MouseMode.mouse_up(self, event)
        self._residue = None
        self._last_y = None
    
    def wheel(self, event):
        r = self._picked_residue(event)
        if r:
            d = event.wheel_value()
            rstep = 1 if d > 0 else -1
            self._swap_residue_step(r, rstep)

    def _picked_residue(self, event):
        x,y = event.position()
        from chimerax.ui.mousemodes import picked_object
        pick = picked_object(x, y, self.session.main_view)
        return self._residue_from_pick(pick)

    def _residue_from_pick(self, pick):
        from chimerax.core.atomic import PickedAtom, PickedBond, PickedResidue
        if isinstance(pick, PickedAtom):
            r = pick.atom.residue
        elif isinstance(pick, PickedBond):
            a1,a2 = pick.bond.atoms
            r = a1.residue if a1.residue == a2.residue else None
        elif isinstance(pick, PickedResidue):
            r = pick.residue
        else:
            r = None
        return r

    def _swap_residue_step(self, r, rstep):
        if rstep > -1 and rstep < 1:
            return False
        irstep = int(rstep) if rstep > 0 else -int(-rstep)
        tres = self._template_residues
        rname = r.name
        ri = [i for i,rt in enumerate(tres) if rt.name == rname]
        tr = tres[0] if len(ri) == 0 else tres[(ri[0] + irstep) % len(tres)]
        swapped = self._swap_residue(r, tr)
        if swapped:
            self._label(r)
        return swapped

    def _swap_residue(self, r, new_r):
        pos, amap = self._backbone_alignment(r, new_r)
        if pos is None:
            return False	# Missing backbone atoms to align new residue

        add_hydrogens = self._has_hydrogens(r)
        carbon_color = self._carbon_color(r)

        # Delete atoms.  Backbone atom HA is deleted if new residues is GLY.
        akeep = set(self._keep_atom_names).intersection(new_r.atoms.names)
        from chimerax.core.atomic import Atoms
        adel = Atoms([a for a in r.atoms if a.name not in akeep])
        adel.delete()

        # Create new atoms
        s = r.structure
        akept = set(r.atoms.names)
        # Set new atom b-factors to average of previous residue backbone atom b-factors.
        bbf = [a.bfactor for a in r.atoms]
        bfactor = sum(bbf)/len(bbf) if bbf else 0
        from chimerax.core.atomic.colors import element_color
        for a in new_r.atoms:
            if a.name not in akept:
                if a.element.name != 'H' or add_hydrogens:
                    na = s.new_atom(a.name, a.element)
                    na.scene_coord = pos * a.scene_coord
                    # TODO: Color by element, but use model carbon color.
                    na.color = carbon_color if a.element.name == 'C' else element_color(a.element.number)
                    na.draw_mode = na.STICK_STYLE
                    na.bfactor = bfactor
                    r.add_atom(na)
                    amap[a] = na
        
        # Create new bonds
        for b in new_r.atoms.intra_bonds:
            a1,a2 = b.atoms
            if a1 in amap and a2 in amap:
                na1, na2 = amap[a1], amap[a2]
                if not na1.connects_to(na2):
                    nb = s.new_bond(na1, na2)

        # Set new residue name.
        r.name = new_r.name

        return True
    
    def _has_hydrogens(self, r):
        for a in r.atoms:
            if a.element.name == 'H':
                return True
        return False

    def _carbon_color(self, r):
        for a in r.atoms:
            if a.element.name == 'C':
                return a.color
        from chimerax.core.atomic.colors import element_color
        return element_color(6)
    
    def _has_alignment_atoms(self, r):
        aan = self._align_atom_names
        if len([a for a in r.atoms if a.name in aan]) < 3:
            log = self.session.logger
            log.status('swapaa cannot align to residues %s which has fewer than 3 backbone atoms (%s)'
                       % (str(r), ', '.join(aan)), log = True)
            return False
        return True
        
    def _backbone_alignment(self, r, new_r):
        ra = dict((a.name, a) for a in r.atoms)
        nra = dict((a.name, a) for a in new_r.atoms)
        apairs = []
        aan = self._align_atom_names
        for aname in aan:
            if aname in ra and aname in nra:
                apairs.append((ra[aname], nra[aname]))
        if len(apairs) < 3:
            log = self.session.logger
            log.status('Fewer than 3 backbone atoms (%s) in residue %s (%s), swapaa cannot align'
                       % (', '.join(aan), str(r), ', '.join(ra.keys())), log = True)
            return None, None
        from chimerax.core.geometry import align_points
        from numpy import array
        xyz = array([a2.scene_coord for a1,a2 in apairs])
        ref_xyz = array([a1.scene_coord for a1,a2 in apairs])
        p, rms = align_points(xyz, ref_xyz)
        amap = dict((a2,a1) for a1,a2 in apairs)
        return p, amap

    def _label(self, r):
        from chimerax.label.label3d import label
        rname = r.name
        objects, otype = self._label_objects(r)
        label(self.session, objects, otype, text = rname)

    def _label_objects(self, r):
        # Label CA atom so label does not jump around.
        la = [a for a in r.atoms if a.name == self._label_atom_name]
        from chimerax.core.objects import Objects
        if len(la) == 1:
            from chimerax.core.atomic import Atoms
            objects = Objects(atoms = Atoms(la))
            otype = 'atoms'
        else:
            # If no CA atom them label center of residue
            objects = Objects(atoms = r.atoms)
            otype = 'residues'
        return objects, otype
    
    def _unlabel(self):
        r = self._residue
        if r is not None:
            objects, otype = self._label_objects(r)
            from chimerax.label.label3d import label_delete
            label_delete(self.session, objects, otype)
        
    def laser_click(self, xyz1, xyz2):
        from chimerax.ui.mousemodes import picked_object_on_segment
        pick = picked_object_on_segment(xyz1, xyz2, self.view)
        r = self._residue_from_pick(pick)
        if r and self._has_alignment_atoms(r):
            self._residue = r

    def drag_3d(self, position, move, delta_z):
        if delta_z is None:
            self._unlabel()
            self._residue = None
        else:
            r = self._residue
            if r:
                rstep = delta_z / self._step_meters
                if not self._swap_residue_step(r, rstep):
                    return 'accumulate drag'
