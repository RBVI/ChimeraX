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

from .molobject import Atom, Bond, Chain, CoordSet, Element, Pseudobond, Residue, Sequence, \
    StructureSeq, PseudobondManager, Ring, ChangeTracker
from .molobject import SeqMatchMap, estimate_assoc_params, try_assoc, StructAssocError
# pbgroup must precede molarray since molarray uses interatom_pseudobonds in global scope
from .pbgroup import PseudobondGroup, all_pseudobond_groups, interatom_pseudobonds, selected_pseudobonds
from .molarray import Collection, Atoms, AtomicStructures, Bonds, Chains, Pseudobonds, \
    PseudobondGroups, Residues, concatenate
from .structure import AtomicStructure, Structure, LevelOfDetail
from .structure import selected_atoms, selected_bonds, selected_residues
from .structure import all_atoms, all_atomic_structures, all_structures
from .structure import structure_atoms, structure_residues, structure_graphics_updater, level_of_detail
from .structure import PickedAtom, PickedBond, PickedResidue, PickedPseudobond
from .molsurf import buried_area, MolecularSurface, surfaces_with_atoms
from .changes import check_for_changes
from .pdbmatrices import biological_unit_matrices
from .triggers import get_triggers
from .shapedrawing import AtomicShapeDrawing
from .args import SymmetryArg, AtomsArg, ResiduesArg, UniqueChainsArg, AtomicStructuresArg
from .args import StructureArg, StructuresArg
from .args import BondArg, BondsArg, PseudobondsArg, PseudobondGroupsArg
from .cytmpl import TmplResidue


from chimerax.core.toolshed import BundleAPI

class _AtomicBundleAPI(BundleAPI):

    KNOWN_CLASSES = {
        "Atom", "AtomicStructure", "AtomicStructures", "Atoms", "Bond", "Bonds",
        "Chain", "Chains", "CoordSet", "LevelOfDetail", "MolecularSurface",
        "PseudobondGroup", "PseudobondGroups", "PseudobondManager", "Pseudobond", "Pseudobonds",
        "Residue", "Residues", "SeqMatchMap", "Sequence", "Structure", "StructureSeq",
        "AtomicShapeDrawing",
    }

    @staticmethod
    def get_class(class_name):
        if class_name in _AtomicBundleAPI.KNOWN_CLASSES:
            import importlib
            this_mod = importlib.import_module(".", __package__)
            return getattr(this_mod, class_name)
        elif class_name in ["AttrRegistration", "CustomizedInstanceManager", "_NoDefault",
                "RegAttrManager"]:
            from . import attr_registration
            return getattr(attr_registration, class_name)
        elif class_name == "XSectionManager":
            from . import ribbon
            return ribbon.XSectionManager

    @staticmethod
    def initialize(session, bundle_info):
        from . import settings
        settings.settings = settings._AtomicSettings(session, "atomic")

        Residue.set_templates_dir(bundle_info.data_dir())

        session.change_tracker = ChangeTracker()
        session.pb_manager = PseudobondManager(session)

        from . import attr_registration
        session.attr_registration = attr_registration.RegAttrManager()
        session.custom_attr_preserver = attr_registration.CustomizedInstanceManager()

        session._atomic_command_handler = session.triggers.add_handler("command finished",
            lambda *args: check_for_changes(session))

        if session.ui.is_gui:
            session.ui.triggers.add_handler('ready', lambda *args, ses=session:
                _AtomicBundleAPI._add_gui_items(ses))
            session.ui.triggers.add_handler('ready', lambda *args, ses=session:
                settings.register_settings_options(ses))

    @staticmethod
    def run_provider(session, bundle_info, name, mgr, **kw):
        from .presets import run_preset
        run_preset(session, bundle_info, name, mgr, **kw)

    @staticmethod
    def finish(session, bundle_info):
        session.triggers.remove_handler(session._atomic_command_handler)

    @staticmethod
    def register_selector(selector_name, logger):
        # 'register_selector' is lazily called when selector is referenced
        from .selectors import register_selectors
        register_selectors(logger)

    @staticmethod
    def _add_gui_items(session):
        from .selectors import add_select_menu_items
        add_select_menu_items(session)

        from .contextmenu import add_selection_context_menu_items
        add_selection_context_menu_items(session)
        
bundle_api = _AtomicBundleAPI()
