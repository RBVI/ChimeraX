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
class SwapAAMouseMode(MouseMode):
    name = 'swapaa'
    icon_file = 'swapaa.png'

    def __init__(self, session):
        MouseMode.__init__(self, session)
        self._residue = None
        self._align_atom_names = ['N', 'C', 'CA']
        self._step_pixels = 20
        self._step_meters = 0.05
        self._last_y = None
        self._label_atom_name = 'CA'		# Which atom to show residue label on.
        
    def enable(self):
        from . import swapaa
        swapaa.template_residues(self.session)
        
    def mouse_down(self, event):
        MouseMode.mouse_down(self, event)
        r = self._picked_residue(event)
        if r and self._has_alignment_atoms(r):
            self._residue = r
            self._last_y = event.position()[1]
    
    def mouse_drag(self, event):
        r = self._residue
        if r and not r.deleted:
            x,y = event.position()
            dy = y - self._last_y
            if abs(dy) >= self._step_pixels:
                rstep = dy / self._step_pixels
                if self._swap_residue_step(r, rstep):
                    self._last_y = y

    def mouse_up(self, event):
        self._unlabel()
        MouseMode.mouse_up(self, event)
        r = self._residue
        self._residue = None
        self._last_y = None
        _log_swapaa_command(r)
    
    def wheel(self, event):
        r = self._picked_residue(event)
        if r:
            d = event.wheel_value()
            rstep = 1 if d > 0 else -1
            self._swap_residue_step(r, rstep)

    def _picked_residue(self, event):
        x,y = event.position()
        pick = self.session.main_view.picked_object(x, y)
        return self._residue_from_pick(pick)

    def _residue_from_pick(self, pick):
        from chimerax.atomic import PickedAtom, PickedBond, PickedResidue
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
        from . import swapaa
        tres = swapaa.template_residues(self.session).list()
        rname = r.name
        ri = [i for i,rt in enumerate(tres) if rt.name == rname]
        tr = tres[0] if len(ri) == 0 else tres[(ri[0] + irstep) % len(tres)]
        swapped = swapaa.swap_residue(r, tr, align_atom_names = self._align_atom_names)
        if swapped:
            self._label(r)
        return swapped
    
    def _has_alignment_atoms(self, r):
        aan = self._align_atom_names
        if len([a for a in r.atoms if a.name in aan]) < 3:
            log = self.session.logger
            log.status('swapaa cannot align to residues %s which has fewer than 3 backbone atoms (%s)'
                       % (str(r), ', '.join(aan)), log = True)
            return False
        return True

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
            from chimerax.atomic import Atoms
            objects = Objects(atoms = Atoms(la))
            otype = 'atoms'
        else:
            # If no CA atom them label center of residue
            objects = Objects(atoms = r.atoms)
            otype = 'residues'
        return objects, otype
    
    def _unlabel(self):
        r = self._residue
        if r is not None and not r.deleted:
            objects, otype = self._label_objects(r)
            from chimerax.label.label3d import label_delete
            label_delete(self.session, objects, otype)
        
    def vr_press(self, event):
        # Virtual reality hand controller button press.
        pick = event.picked_object(self.view)
        r = self._residue_from_pick(pick)
        if r and self._has_alignment_atoms(r):
            self._residue = r

    def vr_motion(self, event):
        r = self._residue
        if r and not r.deleted:
            rstep = event.room_vertical_motion / self._step_meters
            if not self._swap_residue_step(r, rstep):
                return 'accumulate drag'

    def vr_release(self, event):
        # Virtual reality hand controller button release.
        self._unlabel()
        r = self._residue
        self._residue = None
        _log_swapaa_command(r)

def _log_swapaa_command(res):
    if res is None or res.deleted:
        return
    ses = res.structure.session
    cmd = 'swapaa mousemode %s %s' % (res.string(style = 'command'), res.name)
    from chimerax.core.commands import run
    run(ses, cmd)
