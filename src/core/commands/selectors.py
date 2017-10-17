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
    logger = session.logger
    reg("sel", _sel_selector, logger)
    reg("all", _all_selector, logger)
    reg("ions", lambda s, m, r: _structure_category_selector("ions", m, r), logger)
    reg("ligand", lambda s, m, r: _structure_category_selector("ligand", m, r), logger)
    reg("main", lambda s, m, r: _structure_category_selector("main", m, r), logger)
    reg("solvent", lambda s, m, r: _structure_category_selector("solvent", m, r), logger)
    reg("strand", _strands_selector, logger)
    reg("helix", _helices_selector, logger)
    reg("coil", _coil_selector, logger)
    reg("protein", lambda s, m, r: _polymer_selector(m, r, True), logger)
    reg("nucleic", lambda s, m, r: _polymer_selector(m, r, False), logger)
    reg("nucleic-acid", lambda s, m, r: _polymer_selector(m, r, False), logger)
    reg("pbonds", _pbonds_selector, logger)
    reg("hbonds", _hbonds_selector, logger)
    reg("backbone", _backbone_selector, logger)
    reg("mainchain", _backbone_selector, logger)
    reg("sidechain", _sidechain_selector, logger)
    reg("sideonly", _sideonly_selector, logger)
    reg("ribose", _ribose_selector, logger)
    from ..atomic import Element, Atom
    # Since IDATM has types in conflict with element symbols (e.g. 'H'), register
    # the types first so that they get overriden by the symbols
    for idatm in Atom.idatm_info_map.keys():
        reg(idatm, lambda ses, models, results, sym=idatm: _idatm_selector(sym, models, results), logger)
    for i in range(1, 115):
        e = Element.get_element(i)
        reg(e.name, lambda ses, models, results, sym=e.name: _element_selector(sym, models, results), logger)

    

def _element_selector(symbol, models, results):
    from ..atomic import Structure
    for m in models:
        if isinstance(m, Structure):
            atoms = m.atoms.filter(m.atoms.element_names == symbol)
            if len(atoms) > 0:
                results.add_model(m)
                results.add_atoms(atoms, bonds=True)

def _idatm_selector(symbol, models, results):
    from ..atomic import Structure
    for m in models:
        if isinstance(m, Structure):
            atoms = m.atoms.filter(m.atoms.idatm_types == symbol)
            if len(atoms) > 0:
                results.add_model(m)
                results.add_atoms(atoms, bonds=True)

def _sel_selector(session, models, results):
    from ..atomic import Structure, PseudobondGroup
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
            for bonds in m.selected_items('bonds'):
                results.add_bonds(bonds)
        if isinstance(m, PseudobondGroup):
            pbonds = m.pseudobonds
            bsel = pbonds.selected
            if bsel.any():
                results.add_pseudobonds(pbonds[bsel])

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
            results.add_atoms(m.atoms, bonds=True)

def _strands_selector(session, models, results):
    from ..atomic import Structure
    for m in models:
        if isinstance(m, Structure):
            strands = m.residues.filter(m.residues.is_strand)
            if strands:
                results.add_model(m)
                results.add_atoms(strands.atoms, bonds=True)

def _structure_category_selector(cat, models, results):
    from ..atomic import Structure
    for m in models:
        if isinstance(m, Structure):
            atoms = m.atoms.filter(m.atoms.structure_categories == cat)
            if len(atoms) > 0:
                results.add_model(m)
                results.add_atoms(atoms, bonds=True)

def _helices_selector(session, models, results):
    from ..atomic import Structure
    for m in models:
        if isinstance(m, Structure):
            helices = m.residues.filter(m.residues.is_helix)
            if helices:
                results.add_model(m)
                results.add_atoms(helices.atoms, bonds=True)

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
                results.add_atoms(coil.atoms, bonds=True)

def _get_missing_structure(struct, atoms):
    pbg = struct.pseudobond_group("missing structure", create_type=None)
    pbs = []
    if pbg:
        for pb in pbg.pseudobonds:
            a1, a2 = pb.atoms
            if a1 in atoms and a2 in atoms:
                pbs.append(pb)
    return pbs, pbg

def _add_missing_structure(results, pbs, pbg):
    from ..atomic import Pseudobonds
    results.add_pseudobonds(Pseudobonds(pbs))
    results.add_model(pbg)

def _polymer_selector(models, results, protein):
    from ..atomic import Structure
    for m in models:
        if isinstance(m, Structure):
            pas = m.residues.existing_principal_atoms
            if protein:
                residues = pas.residues.filter(pas.names=="CA")
            else:
                residues = pas.residues.filter(pas.names!="CA")
            atoms = residues.atoms
            pbs, pbg = _get_missing_structure(m, atoms)
            if residues:
                results.add_model(m)
                results.add_atoms(atoms, bonds=True)
                if pbs:
                    _add_missing_structure(results, pbs, pbg)

def _pbonds_selector(session, models, results):
    from ..atomic import Pseudobonds, PseudobondGroup, concatenate
    pbonds = concatenate([pbg.pseudobonds for pbg in models if isinstance(pbg, PseudobondGroup)],
                         Pseudobonds)
    results.add_pseudobonds(pbonds)
    for m in pbonds.unique_groups:
        results.add_model(m)

def _hbonds_selector(session, models, results):
    from ..atomic import Pseudobonds, PseudobondGroup, concatenate
    pbonds = concatenate([pbg.pseudobonds for pbg in models
                          if isinstance(pbg, PseudobondGroup) and pbg.category == 'hydrogen bonds'],
                         Pseudobonds)
    results.add_pseudobonds(pbonds)
    for m in pbonds.unique_groups:
        results.add_model(m)

def _backbone_selector(session, models, results):
    from ..atomic import Structure, structure_atoms
    atoms = structure_atoms([m for m in models if isinstance(m, Structure)])
    backbone = atoms.filter(atoms.is_backbones())
    if backbone:
        for s, struct_backbone in backbone.by_structure:
            results.add_model(s)
            pbs, pbg = _get_missing_structure(s, struct_backbone)
            if pbs:
                _add_missing_structure(results, pbs, pbg)
        results.add_atoms(backbone, bonds=True)

def _sidechain_selector(session, models, results):
    from ..atomic import Structure, structure_atoms
    atoms = structure_atoms([m for m in models if isinstance(m, Structure)])
    sidechain = atoms.filter(atoms.is_side_chains)
    if sidechain:
        for m in sidechain.unique_structures:
            results.add_model(m)
        results.add_atoms(sidechain, bonds=True)

def _sideonly_selector(session, models, results):
    from ..atomic import Structure, structure_atoms
    atoms = structure_atoms([m for m in models if isinstance(m, Structure)])
    sideonly = atoms.filter(atoms.is_side_onlys)
    if sideonly:
        for m in sideonly.unique_structures:
            results.add_model(m)
        results.add_atoms(sideonly, bonds=True)

def _ribose_selector(session, models, results):
    from ..atomic import Structure, structure_atoms
    atoms = structure_atoms([m for m in models if isinstance(m, Structure)])
    ribose = atoms.filter(atoms.is_riboses)
    if ribose:
        for m in ribose.unique_structures:
            results.add_model(m)
        results.add_atoms(ribose, bonds=True)
