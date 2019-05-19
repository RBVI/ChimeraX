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


name_mapping = {
    'Sticks': 'non-polymer',
    'Cartoon': 'small polymer',
    'Space-Filling (chain colors)': 'medium polymer',
    'Space-Filling (single color)': 'large polymer'
}


def run_preset(session, bundle_info, name, mgr, **kw):
    mgr.execute(lambda session=session, name=name: _execute(session, name))


def _execute(session, name):
    from . import AtomicStructure, Atom, MolecularSurface
    structures = [m for m in session.models if isinstance(m, AtomicStructure)]
    kw = {'set_lighting': len(structures) < 2}
    if name in name_mapping:
        kw['style'] = name_mapping[name]
    from .nucleotides.cmd import nucleotides
    nucleotides(session, 'atoms')
    surfaces = [cm for s in structures
                   for cm in s.child_models()
                   if isinstance(cm, MolecularSurface)]
    for srf in surfaces:
        srf.display = False
    for s in structures:
        atoms = s.atoms
        atoms.displays = True
        atoms.draw_modes = Atom.SPHERE_STYLE
        residues = s.residues
        residues.ribbon_displays = False
        residues.ring_displays = False
        s.apply_auto_styling(**kw)
        #TODO: reset pseudobond groups
