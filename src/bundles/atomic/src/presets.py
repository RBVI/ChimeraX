# vim: set expandtab shiftwidth=4 softtabstop=4:

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

from chimerax.core.utils import CustomSortString

def register_presets(session):
    name_mapping = {
        CustomSortString('Stick', sort_val=1): 'non-polymer',
        CustomSortString('Cartoon', sort_val=2): 'small polymer',
        CustomSortString('Space-Filling (chain colors)', sort_val=3): 'medium polymer',
        CustomSortString('Space-Filling (single color)', sort_val=4): 'large polymer'
    }
    def callback(name, session=session):
        from . import AtomicStructure, Atom
        structures = [m for m in session.models if isinstance(m, AtomicStructure)]
        kw = {'set_lighting': len(structures) < 2}
        if name in name_mapping:
            kw['style'] = name_mapping[name]
        from .nucleotides.cmd import nucleotides
        nucleotides(session, 'atoms')
        for s in structures:
            atoms = s.atoms
            atoms.displays = True
            atoms.draw_modes = Atom.SPHERE_STYLE
            residues = s.residues
            residues.ribbon_displays = False
            residues.ring_displays = False
            s.apply_auto_styling(**kw)
    session.presets.add_presets("Initial Styles", [ (name, lambda nm=name: callback(nm))
        for name in [CustomSortString('Original Look', sort_val=0)] + sorted(list(name_mapping.keys()))])

    # Elements / IDATM Selection menu items
