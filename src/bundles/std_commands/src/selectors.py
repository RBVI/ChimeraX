# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

def register_selectors(logger):
    # Selectors
    #
    # NOTE: also need to be listed in bundle_info.xml.in
    from chimerax.core.commands import register_selector as reg
    reg("sel", _sel_selector, logger, desc="selected atoms")
    reg("all", _all_selector, logger, desc="everything")
    reg("pbonds", _pbonds_selector, logger, desc="pseudobonds")
    reg("pbondatoms", _pbondatoms_selector, logger, desc="pseudobond atoms")
    reg("hbonds", _hbonds_selector, logger, desc="hydrogen bonds")
    reg("hbondatoms", _hbondatoms_selector, logger, desc="hydrogen bond atoms")

def _sel_selector(session, models, results):
    from chimerax.atomic import Structure, PseudobondGroup, selected_atoms
    if len([m for m in models if isinstance(m, Structure)]) == len(session.models.list(type=Structure)):
        per_structure_atoms = False
        results.add_atoms(selected_atoms(session))
    else:
        per_structure_atoms = True
    for m in models:
        if m.selected:
            results.add_model(m)
            spos = m.selected_positions
            if spos is not None and spos.sum() > 0:
                results.add_model_instances(m, spos)
        elif _nonmodel_child_selected(m):
            results.add_model(m)
        if isinstance(m, Structure):
            if per_structure_atoms:
                for atoms in m.selected_items('atoms'):
                    results.add_atoms(atoms)
            for bonds in m.selected_items('bonds'):
                results.add_bonds(bonds)
        if isinstance(m, PseudobondGroup):
            pbonds = m.pseudobonds
            bsel = pbonds.selected
            if bsel.any():
                results.add_pseudobonds(pbonds[bsel])

def _nonmodel_child_selected(m):
    from chimerax.core.models import Model
    for d in m.child_drawings():
        if not isinstance(d, Model):
            if d.highlighted or _nonmodel_child_selected(d):
                return True
    return False

def _all_selector(session, models, results):
    from chimerax.atomic import Structure
    for m in models:
        results.add_model(m)
        if isinstance(m, Structure):
            results.add_atoms(m.atoms, bonds=True)

def _pbonds_selector(session, models, results):
    from chimerax.atomic import Pseudobonds, PseudobondGroup, concatenate
    pbonds = concatenate([pbg.pseudobonds for pbg in models if isinstance(pbg, PseudobondGroup)],
                         Pseudobonds)
    results.add_pseudobonds(pbonds)
    for m in pbonds.unique_groups:
        results.add_model(m)

def _pbondatoms_selector(session, models, results):
    from chimerax.atomic import all_pseudobonds, concatenate
    pbonds = all_pseudobonds(session)
    if len(pbonds) > 0:
        atoms = concatenate(pbonds.atoms)
        results.add_atoms(atoms)
        for m in atoms.unique_structures:
            results.add_model(m)

def _hbonds_selector(session, models, results):
    from chimerax.atomic import Pseudobonds, PseudobondGroup, concatenate
    pbonds = concatenate([pbg.pseudobonds for pbg in models
                          if isinstance(pbg, PseudobondGroup) and pbg.name == 'hydrogen bonds'],
                         Pseudobonds)
    results.add_pseudobonds(pbonds)
    for m in pbonds.unique_groups:
        results.add_model(m)

def _hbondatoms_selector(session, models, results):
    from chimerax.atomic import Pseudobonds, PseudobondGroup, concatenate
    pbonds = concatenate([pbg.pseudobonds for pbg in models
              if isinstance(pbg, PseudobondGroup) and pbg.name == 'hydrogen bonds'], Pseudobonds)
    if len(pbonds) > 0:
        atoms = concatenate(pbonds.atoms)
        results.add_atoms(atoms)
        for m in atoms.unique_structures:
            results.add_model(m)
