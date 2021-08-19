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

# ensure atomic_libs C++ shared libs are linkable by us
import chimerax.atomic_lib

from .molobject import Atom, Bond, Chain, CoordSet, Element, Pseudobond, Residue, Sequence, \
    StructureSeq, PseudobondManager, Ring, ChangeTracker, StructureData
from .molobject import SeqMatchMap, estimate_assoc_params, try_assoc, StructAssocError
from .molobject import next_chain_id, chain_id_characters
# pbgroup must precede molarray since molarray uses interatom_pseudobonds in global scope
from .pbgroup import PseudobondGroup, all_pseudobond_groups, all_pseudobonds
from .pbgroup import interatom_pseudobonds, selected_pseudobonds
from .molarray import Collection, Atoms, AtomicStructures, Bonds, Chains, Pseudobonds, Structures, \
    PseudobondGroups, Residues, concatenate
from .structure import AtomicStructure, Structure, LevelOfDetail
from .structure import selected_atoms, selected_bonds, selected_residues
from .structure import all_atoms, all_bonds, all_residues, all_atomic_structures, all_structures
from .structure import structure_atoms, structure_residues, structure_graphics_updater, level_of_detail
from .structure import PickedAtom, PickedBond, PickedResidue, PickedPseudobond
from .molsurf import buried_area, MolecularSurface, surfaces_with_atoms
from .changes import check_for_changes
from .pdbmatrices import biological_unit_matrices
from .triggers import get_triggers
from .shapedrawing import AtomicShapeDrawing, AtomicShapeInfo
from .args import SymmetryArg, AtomArg, AtomsArg, ResiduesArg, UniqueChainsArg, AtomicStructuresArg
from .args import StructureArg, StructuresArg, ElementArg, OrderedAtomsArg
from .args import BondArg, BondsArg, PseudobondsArg, PseudobondGroupsArg, concise_residue_spec
from .cytmpl import TmplResidue

def initialize_atomic(session):
    from . import settings
    settings.settings = settings._AtomicSettings(session, "atomic")

    from chimerax import atomic_lib
    import os.path
    res_templates_dir = os.path.join(atomic_lib.__path__[0], 'data')
    Residue.set_templates_dir(res_templates_dir)
    
    import chimerax
    if hasattr(chimerax, 'app_dirs'):
        Residue.set_user_templates_dir(chimerax.app_dirs.user_data_dir)

    session.change_tracker = ChangeTracker()
    session.pb_manager = PseudobondManager(session)

    session._atomic_command_handler = session.triggers.add_handler("command finished",
        lambda *args: check_for_changes(session))

    if session.ui.is_gui:
        session.ui.triggers.add_handler('ready', lambda *args, ses=session:
            _AtomicBundleAPI._add_gui_items(ses))
        session.ui.triggers.add_handler('ready', lambda *args, ses=session:
            settings.register_settings_options(ses))


from chimerax.core.toolshed import BundleAPI

class _AtomicBundleAPI(BundleAPI):

    KNOWN_CLASSES = {
        "Atom", "AtomicStructure", "AtomicShapeDrawing", "AtomicStructures", "Atoms", "Bond", "Bonds",
        "Chain", "Chains", "CoordSet", "LevelOfDetail", "MolecularSurface",
        "PseudobondGroup", "PseudobondGroups", "PseudobondManager", "Pseudobond", "Pseudobonds",
        "Residue", "Residues", "SeqMatchMap", "Sequence", "Structure", "StructureSeq",
    }

    @staticmethod
    def get_class(class_name):
        if class_name in _AtomicBundleAPI.KNOWN_CLASSES:
            import importlib
            this_mod = importlib.import_module(".", __package__)
            return getattr(this_mod, class_name)
        elif class_name in ["AttrRegistration", "CustomizedInstanceManager", "_NoDefault",
                "RegAttrManager"]:
            # attribute registration used to be here instead of core, so for session compatibility...
            if class_name == "_NoDefault":
                from chimerax.core.attributes import _NoDefault
                return _NoDefault
            from chimerax.core.session import State
            class Fake(State):
                def reset_state(self, session):
                    pass
                def take_snapshot(self, session, flags):
                    return {}
                def restore_snapshot(session, data, name=class_name):
                    if name == "RegAttrManager":
                        from chimerax.core.attributes import MANAGER_NAME
                        session.get_state_manager(MANAGER_NAME)._restore_session_data(session, data)
                    def remove_fakes(*args, ses=session, fake=Fake):
                        for attr_name in dir(session):
                            if isinstance(getattr(session, attr_name), fake):
                                delattr(session, attr_name)
                        from chimerax.core.triggerset import DEREGISTER
                        return DEREGISTER
                    session.triggers.add_handler('end restore session', remove_fakes)
                    return Fake()
            return Fake
        elif class_name == "XSectionManager":
            from . import ribbon
            return ribbon.XSectionManager
        elif class_name in ["GenericSeqFeature", "SeqVariantFeature"]:
            from . import seq_support
            return getattr(seq_support, class_name)

    @staticmethod
    def initialize(session, bundle_info):
        initialize_atomic(session)

    @staticmethod
    def run_provider(session, name, mgr, **kw):
        from .presets import run_preset
        run_preset(session, name, mgr, **kw)

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
