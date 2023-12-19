# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This particular file is part of the ChimeraX library. You can also
# redistribute and/or modify it under the terms of the GNU Lesser General
# Public License version 2.1 as published by the Free Software Foundation.
# For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
# EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
# LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
# VERSION 2.1
#
# This notice must be embedded in or attached to all copies, including partial
# copies, of the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===


name_mapping = {
    'Sticks': 'non-polymer',
    'Cartoon': 'small polymer',
    'Space-Filling (chain colors)': 'medium polymer',
    'Space-Filling (single color)': 'large polymer'
}


def run_preset(session, name, mgr, **kw):
    mgr.execute(lambda session=session, name=name: _execute(session, name))


def _execute(session, name):
    from . import AtomicStructure, Atom, MolecularSurface
    structures = [m for m in session.models if isinstance(m, AtomicStructure)]
    kw = {'set_lighting': len(structures) < 2}
    if name in name_mapping:
        kw['style'] = name_mapping[name]
    from chimerax.nucleotides.cmd import nucleotides
    nucleotides(session, 'atoms', create_undo=False)
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
