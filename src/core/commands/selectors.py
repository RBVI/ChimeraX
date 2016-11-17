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

def register_core_selectors(session):
    # Selectors
    from .atomspec import register_selector as reg
    reg("sel", _sel_selector)
    reg("all", _all_selector)
    reg("ions", lambda s, m, r: _structure_category_selector("ions", m, r))
    reg("ligand", lambda s, m, r: _structure_category_selector("ligand", m, r))
    reg("main", lambda s, m, r: _structure_category_selector("main", m, r))
    reg("solvent", lambda s, m, r: _structure_category_selector("solvent", m, r))
    reg("strand", _strands_selector)
    reg("helix", _helices_selector)
    reg("coil", _coil_selector)
    reg("protein", lambda s, m, r: _polymer_selector(m, r, True))
    reg("nucleic", lambda s, m, r: _polymer_selector(m, r, False))
    reg("nucleic-acid", lambda s, m, r: _polymer_selector(m, r, False))
    reg("pbonds", _pbonds_selector)
    reg("hbonds", _hbonds_selector)
    reg("backbone", _backbone_selector)
    reg("mainchain", _backbone_selector)
    reg("sidechain", _sidechain_selector)
    reg("ribose", _ribose_selector)
    from ..atomic import Element, Atom
    # Since IDATM has types in conflict with element symbols (e.g. 'H'), register
    # the types first so that they get overriden by the symbols
    for idatm in Atom.idatm_info_map.keys():
        reg(idatm, lambda ses, models, results, sym=idatm: _idatm_selector(sym, models, results))
    for i in range(1, 115):
        e = Element.get_element(i)
        reg(e.name, lambda ses, models, results, sym=e.name: _element_selector(sym, models, results))

    

def _element_selector(symbol, models, results):
    from ..atomic import Structure
    for m in models:
        if isinstance(m, Structure):
            atoms = m.atoms.filter(m.atoms.element_names == symbol)
            if len(atoms) > 0:
                results.add_model(m)
                results.add_atoms(atoms)

def _idatm_selector(symbol, models, results):
    from ..atomic import Structure
    for m in models:
        if isinstance(m, Structure):
            atoms = m.atoms.filter(m.atoms.idatm_types == symbol)
            if len(atoms) > 0:
                results.add_model(m)
                results.add_atoms(atoms)

def _sel_selector(session, models, results):
    from ..atomic import Structure
    for m in models:
        if m.selected:
            results.add_model(m)
            spos = m.selected_positions
            if spos is not None and spos.sum() > 0:
                results.add_model_instances(m, spos)
        elif _nonmodel_child_selected(m):
            results.add_model(m)
        if isinstance(m, Structure):
            for atoms in m.selected_items('atoms'):
                results.add_atoms(atoms)

def _nonmodel_child_selected(m):
    from ..models import Model
    for d in m.child_drawings():
        if not isinstance(d, Model):
            if d.selected or _nonmodel_child_selected(d):
                return True
    return False

def _all_selector(session, models, results):
    from ..atomic import Structure
    for m in models:
        results.add_model(m)
        if isinstance(m, Structure):
            results.add_atoms(m.atoms)

def _strands_selector(session, models, results):
    from ..atomic import Structure
    for m in models:
        if isinstance(m, Structure):
            strands = m.residues.filter(m.residues.is_strand)
            if strands:
                results.add_model(m)
                results.add_atoms(strands.atoms)

def _structure_category_selector(cat, models, results):
    from ..atomic import Structure
    for m in models:
        if isinstance(m, Structure):
            atoms = m.atoms.filter(m.atoms.structure_categories == cat)
            if len(atoms) > 0:
                results.add_model(m)
                results.add_atoms(atoms)

def _helices_selector(session, models, results):
    from ..atomic import Structure
    for m in models:
        if isinstance(m, Structure):
            helices = m.residues.filter(m.residues.is_helix)
            if helices:
                results.add_model(m)
                results.add_atoms(helices.atoms)

def _coil_selector(session, models, results):
    from ..atomic import Structure
    for m in models:
        if isinstance(m, Structure):
            from numpy import logical_not, logical_or
            cr = m.chains.existing_residues
            is_coil = logical_not(logical_or(cr.is_strand, cr.is_helix))
            coil = cr.filter(is_coil)
            # also exclude nucleic acids
            coil = coil.existing_principal_atoms.residues
            coil = coil.filter(coil.existing_principal_atoms.names == "CA")
            if coil:
                results.add_model(m)
                results.add_atoms(coil.atoms)

def _polymer_selector(models, results, protein):
    from ..atomic import Structure
    for m in models:
        if isinstance(m, Structure):
            pas = m.residues.existing_principal_atoms
            if protein:
                residues = pas.residues.filter(pas.names=="CA")
            else:
                residues = pas.residues.filter(pas.names!="CA")
            if residues:
                results.add_model(m)
                results.add_atoms(residues.atoms)

def _pbonds_selector(session, models, results):
    from ..atomic import Structure, structure_atoms, interatom_pseudobonds
    atoms = structure_atoms([m for m in models if isinstance(m, Structure)])
    pbonds = interatom_pseudobonds(atoms)
    a1, a2 = pbonds.atoms
    atoms = a1 | a2
    for m in atoms.unique_structures:
        results.add_model(m)
    results.add_atoms(atoms)

def _hbonds_selector(session, models, results):
    from ..atomic import Structure, structure_atoms, interatom_pseudobonds
    atoms = structure_atoms([m for m in models if isinstance(m, Structure)])
    pbonds = interatom_pseudobonds(atoms, group_name = 'hydrogen bonds')
    a1, a2 = pbonds.atoms
    atoms = a1 | a2
    for m in atoms.unique_structures:
        results.add_model(m)
    results.add_atoms(atoms)

def _backbone_selector(session, models, results):
    from ..atomic import Structure, structure_atoms
    atoms = structure_atoms([m for m in models if isinstance(m, Structure)])
    backbone = atoms.filter(atoms.is_backbones())
    if backbone:
        for m in backbone.unique_structures:
            results.add_model(m)
        results.add_atoms(backbone)

def _sidechain_selector(session, models, results):
    from ..atomic import Structure, structure_atoms
    atoms = structure_atoms([m for m in models if isinstance(m, Structure)])
    sidechain = atoms.filter(atoms.is_sidechains)
    if sidechain:
        for m in sidechain.unique_structures:
            results.add_model(m)
        results.add_atoms(sidechain)

def _ribose_selector(session, models, results):
    from ..atomic import Structure, structure_atoms
    atoms = structure_atoms([m for m in models if isinstance(m, Structure)])
    ribose = atoms.filter(atoms.is_riboses)
    if ribose:
        for m in ribose.unique_structures:
            results.add_model(m)
        results.add_atoms(ribose)
