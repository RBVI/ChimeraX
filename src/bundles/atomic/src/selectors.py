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

def register_selectors(logger):
    # Selectors
    #
    # NOTE: also need to be listed in bundle_info.xml.in
    from chimerax.core.commands import register_selector as reg
    from . import Element, Atom
    # Since IDATM has types in conflict with element symbols (e.g. 'H'), register
    # the types first so that they get overriden by the symbols
    for idatm, info in Atom.idatm_info_map.items():
        reg(idatm, lambda ses, models, results, sym=idatm: _idatm_selector(sym, models, results), logger, desc=info.description)
    for i in range(1, Element.NUM_SUPPORTED_ELEMENTS):
        e = Element.get_element(i)
        reg(e.name, lambda ses, models, results, sym=e.name: _element_selector(sym, models, results), logger, desc="%s (element)" % e.name)

    

def _element_selector(symbol, models, results):
    from chimerax.atomic import Structure
    for m in models:
        if isinstance(m, Structure):
            atoms = m.atoms.filter(m.atoms.element_names == symbol)
            if len(atoms) > 0:
                results.add_model(m)
                results.add_atoms(atoms, bonds=True)

def _idatm_selector(symbol, models, results):
    from chimerax.atomic import Structure
    for m in models:
        if isinstance(m, Structure):
            atoms = m.atoms.filter(m.atoms.idatm_types == symbol)
            if len(atoms) > 0:
                results.add_model(m)
                results.add_atoms(atoms, bonds=True)
