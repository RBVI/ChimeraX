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
    from . import Element, Atom
    # Since IDATM has types in conflict with element symbols (e.g. 'H'), register
    # the types first so that they get overriden by the symbols
    for idatm, info in Atom.idatm_info_map.items():
        reg(idatm, lambda ses, models, results, sym=idatm: _idatm_selector(sym, models, results), logger, desc=info.description)
    for i in range(1, Element.NUM_SUPPORTED_ELEMENTS):
        e = Element.get_element(i)
        reg(e.name, lambda ses, models, results, sym=e.name: _element_selector(sym, models, results), logger, desc="%s (element)" % e.name)
    reg("backbone", _backbone_selector, logger, desc="backbone atoms")
    reg("mainchain", _backbone_selector, logger, desc="backbone atoms")
    reg("min-backbone", _min_backbone_selector, logger, desc="minimal backbone atoms")
    reg("protein", lambda s, m, r: _polymer_selector(m, r, True), logger, desc="proteins")
    reg("nucleic", lambda s, m, r: _polymer_selector(m, r, False), logger, desc="nucleic acids")
    reg("nucleic-acid", lambda s, m, r: _polymer_selector(m, r, False), logger, desc="nuecleic acids")
    reg("ions", lambda s, m, r: _structure_category_selector("ions", m, r), logger, desc="ions")
    reg("ligand", lambda s, m, r: _structure_category_selector("ligand", m, r), logger, desc="ligands")
    reg("main", lambda s, m, r: _structure_category_selector("main", m, r), logger, desc="main structure")
    reg("sel-residues", _sel_residues, logger, desc="current selection promoted to full residues")
    reg("solvent", lambda s, m, r: _structure_category_selector("solvent", m, r), logger, desc="solvent")
    reg("strand", _strands_selector, logger, desc="strands")
    reg("helix", _helices_selector, logger, desc="helices")
    reg("coil", _coil_selector, logger, desc="coils")
    reg("sidechain", _sidechain_selector, logger, desc="side-chain atoms")
    reg("sideonly", _sideonly_selector, logger, desc="side-chain atoms")
    reg("ribose", _ribose_selector, logger, desc="ribose")
    reg("template-mismatch", _missing_heavies, logger, desc="missing heavy atoms")

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

def _backbone_selector(session, models, results):
    from chimerax.atomic import Structure, structure_atoms
    atoms = structure_atoms([m for m in models if isinstance(m, Structure)])
    backbone = atoms.filter(atoms.is_backbones())
    if backbone:
        for s, struct_backbone in backbone.by_structure:
            results.add_model(s)
            pbs, pbg = _get_missing_structure(s, struct_backbone)
            if pbs:
                _add_missing_structure(results, pbs, pbg)
        results.add_atoms(backbone, bonds=True)

def _min_backbone_selector(session, models, results):
    from chimerax.atomic import Structure, structure_atoms, Atom
    atoms = structure_atoms([m for m in models if isinstance(m, Structure)])
    backbone = atoms.filter(atoms.is_backbones(Atom.BBE_MIN))
    if backbone:
        for s, struct_backbone in backbone.by_structure:
            results.add_model(s)
            pbs, pbg = _get_missing_structure(s, struct_backbone)
            if pbs:
                _add_missing_structure(results, pbs, pbg)
        results.add_atoms(backbone, bonds=True)

def _sel_residues(session, models, results):
    from chimerax.atomic import selected_atoms
    results.add_atoms(selected_atoms(session).residues.unique().atoms)

def _polymer_selector(models, results, protein):
    from chimerax.atomic import Structure, Residue
    for m in models:
        if isinstance(m, Structure):
            residues = m.residues.filter(
                m.residues.polymer_types == (Residue.PT_PROTEIN if protein else Residue.PT_NUCLEIC))
            atoms = residues.atoms
            pbs, pbg = _get_missing_structure(m, atoms)
            if residues:
                results.add_model(m)
                results.add_atoms(atoms, bonds=True)
                if pbs:
                    _add_missing_structure(results, pbs, pbg)

def _add_missing_structure(results, pbs, pbg):
    from chimerax.atomic import Pseudobonds
    results.add_pseudobonds(Pseudobonds(pbs))
    results.add_model(pbg)

def _structure_category_selector(cat, models, results):
    from chimerax.atomic import AtomicStructure
    for m in models:
        if isinstance(m, AtomicStructure):
            atoms = m.atoms.filter(m.atoms.structure_categories == cat)
            if len(atoms) > 0:
                results.add_model(m)
                results.add_atoms(atoms, bonds=True)

def _get_missing_structure(struct, atoms):
    pbg = struct.pseudobond_group("missing structure", create_type=None)
    pbs = []
    ptr_set = set(atoms.pointers)
    if pbg:
        for pb in pbg.pseudobonds:
            a1, a2 = pb.atoms
            if a1.cpp_pointer in ptr_set and a2.cpp_pointer in ptr_set:
                pbs.append(pb)
    return pbs, pbg

def _strands_selector(session, models, results):
    from chimerax.atomic import Structure
    for m in models:
        if isinstance(m, Structure):
            strands = m.residues.filter(m.residues.is_strand)
            if strands:
                results.add_model(m)
                results.add_atoms(strands.atoms, bonds=True)

def _helices_selector(session, models, results):
    from chimerax.atomic import Structure
    for m in models:
        if isinstance(m, Structure):
            helices = m.residues.filter(m.residues.is_helix)
            if helices:
                results.add_model(m)
                results.add_atoms(helices.atoms, bonds=True)

def _coil_selector(session, models, results):
    from chimerax.atomic import Structure
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

def _sidechain_selector(session, models, results):
    from chimerax.atomic import Structure, structure_atoms
    atoms = structure_atoms([m for m in models if isinstance(m, Structure)])
    sidechain = atoms.filter(atoms.is_side_chains)
    if sidechain:
        for m in sidechain.unique_structures:
            results.add_model(m)
        results.add_atoms(sidechain, bonds=True)

def _sideonly_selector(session, models, results):
    from chimerax.atomic import Structure, structure_atoms
    atoms = structure_atoms([m for m in models if isinstance(m, Structure)])
    sideonly = atoms.filter(atoms.is_side_onlys)
    if sideonly:
        for m in sideonly.unique_structures:
            results.add_model(m)
        results.add_atoms(sideonly, bonds=True)

def _ribose_selector(session, models, results):
    from chimerax.atomic import Structure, structure_atoms
    atoms = structure_atoms([m for m in models if isinstance(m, Structure)])
    ribose = atoms.filter(atoms.is_riboses)
    if ribose:
        for m in ribose.unique_structures:
            results.add_model(m)
        results.add_atoms(ribose, bonds=True)

def _missing_heavies(session, models, results):
    from chimerax.atomic import Structure, structure_residues
    residues = structure_residues([m for m in models if isinstance(m, Structure)])
    missing = residues.filter(residues.is_missing_heavy_template_atoms)
    if missing:
        for m in missing.unique_structures:
            results.add_model(m)
        results.add_atoms(missing.atoms, bonds=True)

_chains_menu_needs_update = False
_chains_menu_name = "&Chains"
_residues_menu_needs_update = False
_residues_menu_name = "&Residues"


def add_select_menu_items(session):
    mw = session.ui.main_window

    parent_menus = [_chains_menu_name]
    select_chains_menu = mw.add_select_submenu(parent_menus[:-1], parent_menus[-1])
    select_chains_menu.aboutToShow.connect(lambda *, ses=session: _update_select_chains_menu(ses))
    select_chains_menu.setToolTipsVisible(True)
    from . import get_triggers
    atom_triggers = get_triggers()
    atom_triggers.add_handler("changes", _check_chains_update_status)

    from .widgets import make_elements_menu
    elements_menu = make_elements_menu(mw, _session=session, _parent_menus=["Che&mistry", "&Element"])
    elements_menu.triggered.connect(lambda act, mw=mw: mw.select_by_mode(act.text()))

    parent_menus = ["Che&mistry", "&IDATM Type"]
    idatm_menu = mw.add_select_submenu(parent_menus[:-1], parent_menus[-1])
    idatm_menu.triggered.connect(lambda act, mw=mw: mw.select_by_mode(act.text()))
    from Qt.QtGui import QAction
    from . import Atom
    for idatm in Atom.idatm_info_map.keys():
        idatm_menu.addAction(QAction(idatm, mw))

    parent_menus = [_residues_menu_name]
    select_residues_menu = mw.add_select_submenu(parent_menus[:-1], parent_menus[-1])
    select_residues_menu.aboutToShow.connect(lambda *, ses=session: _update_select_residues_menu(ses))
    select_residues_menu.setToolTipsVisible(True)
    from . import get_triggers
    atom_triggers = get_triggers()
    atom_triggers.add_handler("changes", _check_residues_update_status)

    parent_menus = ["&Structure"]
    select_structure_menu = mw.add_select_submenu(parent_menus[:-1], parent_menus[-1])
    select_structure_menu.addAction(QAction("Backbone", mw))
    select_structure_menu.addAction(QAction("Ions", mw))
    select_structure_menu.addAction(QAction("Ligand", mw))
    select_structure_menu.addAction(QAction("Main", mw))
    # Nucleic Acid/Protein/Ribose move to Chemistry menu (chem_group bundle)
    parent_menus = ["&Structure", "&Secondary Structure"]
    ss_menu = mw.add_select_submenu(parent_menus[:-1], parent_menus[-1])
    ss_menu.addAction(QAction("Coil", mw))
    ss_menu.addAction(QAction("Helix", mw))
    ss_menu.addAction(QAction("Strand", mw))
    select_structure_menu.addAction(QAction("Sidechain + Connector", mw))
    select_structure_menu.addAction(QAction("Sidechain Only", mw))
    select_structure_menu.addAction(QAction("Solvent", mw))
    sel_text_remapping = {
        'Sidechain + Connector': 'sidechain',
        'Sidechain Only': 'sideonly'
    }
    select_structure_menu.triggered.connect(lambda act, mw=mw:
        mw.select_by_mode(sel_text_remapping.get(act.text(), act.text()).lower().replace(' ', '-')))

def _update_select_chains_menu(session):
    global _chains_menu_needs_update
    if not _chains_menu_needs_update:
        return
    mw = session.ui.main_window
    select_chains_menu = mw.add_select_submenu([], _chains_menu_name)
    from . import AtomicStructures, all_atomic_structures
    structures = AtomicStructures(all_atomic_structures(session))
    chain_info = {}
    for chain in structures.chains:
        key = chain.description if chain.description else chain.chain_id
        chain_info.setdefault(key, []).append(chain)
    chain_keys = list(chain_info.keys())
    chain_keys.sort()
    select_chains_menu.clear()
    def final_description(description):
        if len(description) < 110:
            return False, description
        return True, description[:50] + "..." + description[-50:]
    from Qt.QtGui import QAction
    for chain_key in chain_keys:
        chains = chain_info[chain_key]
        if len(chains) > 1:
            submenu = select_chains_menu.addMenu(chain_key)
            sep = submenu.addSeparator()
            chains.sort(key=lambda c: (c.structure.id, c.chain_id))
            collective_spec = ""
            for chain in chains:
                if len(structures) > 1:
                    if chain.description:
                        shortened, final = final_description(chain.description)
                        label = "[%s] chain %s" % (chain.structure, chain.chain_id)
                    else:
                        label = "[%s]" % chain.structure
                        shortened = False
                else:
                    # ...must be multiple identical descriptions...
                    label = "Chain %s" % chain.chain_id
                    shortened = False
                spec = chain.string(style="command")
                action = mw.add_menu_selector(submenu, label, spec)
                collective_spec += spec
                if shortened:
                    submenu.setToolTipsVisible(True)
                    action.setToolTip(chain.description)
            mw.add_menu_selector(submenu, "All", collective_spec, insertion_point=sep)
        else:
            chain = chains[0]
            chain_id_text = str(chain)
            slash_index = chain_id_text.rfind('/')
            if slash_index > 0:
                chain_id_text = chain_id_text[:slash_index] + " chain " + chain_id_text[slash_index+1:]
            else:
                chain_id_text = "chain " + chain_id_text[slash_index+1:]
            if chain.description:
                shortened, final = final_description(chain.description)
                label = "%s (%s)" % (final, chain_id_text)
            else:
                label = chain_id_text
                shortened = False
            spec = chain.string(style="command")
            action = mw.add_menu_selector(select_chains_menu, label, spec)
            if shortened:
                select_chains_menu.setToolTipsVisible(True)
                action.setToolTip(chain.description)

    _chains_menu_needs_update = False

def _check_chains_update_status(trig_name, changes):
    global _chains_menu_needs_update
    if not _chains_menu_needs_update:
        _chains_menu_needs_update = changes.created_chains() or changes.num_deleted_chains() > 0

def _update_select_residues_menu(session):
    global _residues_menu_needs_update
    if not _residues_menu_needs_update:
        return
    mw = session.ui.main_window
    select_residues_menu = mw.add_select_submenu([], _residues_menu_name)
    select_residues_menu.clear()
    from . import AtomicStructures, all_atomic_structures
    structures = AtomicStructures(all_atomic_structures(session))
    nonstandard = set()
    amino = set()
    nucleic = set()
    from . import Sequence
    for r in structures.residues:
        if r.polymer_type == r.PT_NONE or Sequence.rname3to1(r.name) == 'X':
            nonstandard.add(r.name)
        elif Sequence.amino3to1(r.name) == 'X':
            nucleic.add(r.name)
        else:
            amino.add(r.name)
    prev_entries = False
    for cat_name, members in (("All Nonstandard", nonstandard), ("Standard Nucleic Acids", nucleic),
            ("Standard Amino Acids", amino)):
        if not members:
            continue
        collective_spec = ""
        if prev_entries:
            select_residues_menu.addSeparator()
        # remember first action so that collective spec can be inserted
        first_entry = None
        res_names = list(members)
        res_names.sort()
        for rn in res_names:
            spec = '::name="%s"' % rn
            collective_spec += spec
            action = mw.add_menu_selector(select_residues_menu, rn, spec)
            if first_entry is None:
                first_entry = action
        if len(res_names) > 1:
            mw.add_menu_selector(select_residues_menu, cat_name, collective_spec, insertion_point=first_entry)
        prev_entries = True

    _residues_menu_needs_update = False

def _check_residues_update_status(trig_name, changes):
    global _residues_menu_needs_update
    if not _residues_menu_needs_update:
        _residues_menu_needs_update = changes.created_residues() or changes.num_deleted_residues() > 0

