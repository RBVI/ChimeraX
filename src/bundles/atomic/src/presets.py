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

def add_presets_menu(session):
    # Presets menu
    name_mapping = {
        'Stick': 'non-polymer',
        'Cartoon': 'small polymer',
        'Space-Filling (chain colors)': 'medium polymer',
        'Space-Filling (single color)': 'large polymer'
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
            s.residues.ribbon_displays = False
            s.apply_auto_styling(**kw)
    for label in ['Original Look'] + sorted(list(name_mapping.keys())):
        session.ui.main_window.add_menu_entry(['Presets'], label,
            lambda name=label: callback(name))

    # Elements / IDATM Selection menu items
