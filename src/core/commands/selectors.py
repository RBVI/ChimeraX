# vim: set expandtab shiftwidth=4 softtabstop=4:

def register_core_selectors(session):
    # Selectors
    from .atomspec import register_selector as reg
    reg(None, "sel", _sel_selector)
    reg(None, "all", _all_selector)
    reg(None, "ions", lambda s, m, r: _structure_category_selector("ions", m, r))
    reg(None, "ligand", lambda s, m, r: _structure_category_selector("ligand", m, r))
    reg(None, "main", lambda s, m, r: _structure_category_selector("main", m, r))
    reg(None, "solvent", lambda s, m, r: _structure_category_selector("solvent", m, r))
    reg(None, "strand", _strands_selector)
    reg(None, "helix", _helices_selector)
    reg(None, "coil", _coil_selector)
    reg(None, "protein", lambda s, m, r: _polymer_selector(m, r, True))
    reg(None, "nucleic", lambda s, m, r: _polymer_selector(m, r, False))
    reg(None, "nucleic-acid", lambda s, m, r: _polymer_selector(m, r, False))
    reg(None, "pbonds", _pbonds_selector)
    reg(None, "hbonds", _hbonds_selector)
    reg(None, "backbone", _backbone_selector)
    reg(None, "mainchain", _backbone_selector)
    reg(None, "sidechain", _sidechain_selector)
    from ..atomic import Element
    for i in range(1, 115):
        e = Element.get_element(i)
        reg(None, e.name, lambda ses, models, results, sym=e.name: _element_selector(sym, models, results))

    

def _element_selector(symbol, models, results):
    from ..atomic import Structure
    for m in models:
        if isinstance(m, Structure):
            atoms = m.atoms.filter(m.atoms.element_names == symbol)
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
            strands = m.residues.filter(m.residues.is_sheet)
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
            is_coil = logical_not(logical_or(cr.is_sheet, cr.is_helix))
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
    pbonds = interatom_pseudobonds(atoms, session)
    a1, a2 = pbonds.atoms
    atoms = a1 | a2
    for m in atoms.unique_structures:
        results.add_model(m)
    results.add_atoms(atoms)

def _hbonds_selector(session, models, results):
    from ..atomic import Structure, structure_atoms, interatom_pseudobonds
    atoms = structure_atoms([m for m in models if isinstance(m, Structure)])
    pbonds = interatom_pseudobonds(atoms, session, group_name = 'hydrogen bonds')
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
