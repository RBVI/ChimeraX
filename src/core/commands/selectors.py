# vi: set expandtab shiftwidth=4 softtabstop=4:

def register_core_selectors(session):
    # Selectors
    from .atomspec import register_selector as reg
    reg(None, "sel", _sel_selector)
    reg(None, "ions", lambda s, m, r: _structure_category_selector("ions", m, r))
    reg(None, "ligand", lambda s, m, r: _structure_category_selector("ligand", m, r))
    reg(None, "main", lambda s, m, r: _structure_category_selector("main", m, r))
    reg(None, "solvent", lambda s, m, r: _structure_category_selector("solvent", m, r))
    reg(None, "strand", _strands_selector)
    reg(None, "helix", _helices_selector)
    reg(None, "coil", _turns_selector)
    reg(None, "pbonds", _pbonds_selector)
    from ..atomic import Element
    for i in range(1, 115):
        e = Element.get_element(i)
        reg(None, e.name, lambda ses, models, results, sym=e.name: _element_selector(sym, models, results))

    

def _element_selector(symbol, models, results):
    from ..atomic import AtomicStructure
    for m in models:
        if isinstance(m, AtomicStructure):
            atoms = m.atoms.filter(m.atoms.element_names == symbol)
            if len(atoms) > 0:
                results.add_model(m)
                results.add_atoms(atoms)

def _sel_selector(session, models, results):
    from ..atomic import AtomicStructure
    for m in models:
        if m.any_part_selected():
            results.add_model(m)
            if isinstance(m, AtomicStructure):
                for atoms in m.selected_items('atoms'):
                    results.add_atoms(atoms)


def _strands_selector(session, models, results):
    from ..atomic import AtomicStructure
    for m in models:
        if isinstance(m, AtomicStructure):
            strands = m.residues.filter(m.residues.is_sheet)
            if strands:
                results.add_model(m)
                results.add_atoms(strands.atoms)

def _structure_category_selector(cat, models, results):
    from ..atomic import AtomicStructure
    for m in models:
        if isinstance(m, AtomicStructure):
            atoms = m.atoms.filter(m.atoms.structure_categories == cat)
            if len(atoms) > 0:
                results.add_model(m)
                results.add_atoms(atoms)

def _helices_selector(session, models, results):
    from ..atomic import AtomicStructure
    for m in models:
        if isinstance(m, AtomicStructure):
            helices = m.residues.filter(m.residues.is_helix)
            if helices:
                results.add_model(m)
                results.add_atoms(helices.atoms)

def _turns_selector(session, models, results):
    from ..atomic import AtomicStructure
    for m in models:
        if isinstance(m, AtomicStructure):
            from numpy import logical_not, logical_or
            is_turn = logical_not(logical_or(m.residues.is_sheet, m.residues.is_helix))
            turns = m.residues.filter(is_turn)
            if turns:
                results.add_model(m)
                results.add_atoms(turns.atoms)

def _pbonds_selector(session, models, results):
    from ..atomic import AtomicStructure, structure_atoms, interatom_pseudobonds
    atoms = structure_atoms([m for m in models if isinstance(m, AtomicStructure)])
    pbonds = interatom_pseudobonds(atoms, session)
    a1, a2 = pbonds.atoms
    atoms = a1 | a2
    for m in atoms.unique_structures:
        results.add_model(m)
    results.add_atoms(atoms)
