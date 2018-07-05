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
    StructureSeq, PseudobondManager, ChangeTracker
from .molobject import SeqMatchMap, estimate_assoc_params, try_assoc, StructAssocError
# pbgroup must precede molarray since molarray uses interatom_pseudobonds in global scope
from .pbgroup import PseudobondGroup, all_pseudobond_groups, interatom_pseudobonds, selected_pseudobonds
from .molarray import Collection, Atoms, AtomicStructures, Bonds, Chains, Pseudobonds, \
    PseudobondGroups, Residues, concatenate
from .structure import AtomicStructure, Structure, LevelOfDetail
from .structure import selected_atoms, selected_bonds
from .structure import all_atoms, all_atomic_structures, all_structures
from .structure import structure_atoms, structure_residues, structure_graphics_updater, level_of_detail
from .structure import PickedAtom, PickedBond, PickedResidue, PickedPseudobond
from .molsurf import buried_area, MolecularSurface, surfaces_with_atoms
from .changes import check_for_changes
from .pdbmatrices import biological_unit_matrices
from .triggers import get_triggers
from .search import atom_search_tree
from .shapedrawing import AtomicShapeDrawing
from .args import SymmetryArg, AtomsArg, UniqueChainsArg, AtomicStructuresArg
from .args import StructureArg, StructuresArg
from .args import BondArg, BondsArg, PseudobondsArg, PseudobondGroupsArg


from chimerax.core.toolshed import BundleAPI

class _AtomicBundleAPI(BundleAPI):

    @staticmethod
    def get_class(class_name):
        if class_name in ["Atom", "AtomicStructure", "AtomicStructures", "Atoms", "Bond", "Bonds",
                "Chain", "Chains", "CoordSet", "LevelOfDetail", "MolecularSurface",
                "PseudobondGroup", "PseudobondManager", "Pseudobond", "Pseudobonds", "Residue",
                "Residues", "SeqMatchMap", "Sequence", "Structure", "StructureSeq"]:
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
    def include_dir(bundle_info):
        from os.path import dirname, join
        return join(dirname(__file__), "include")

    @staticmethod
    def initialize(session, bundle_info):
        """Install alignments manager into existing session"""

        #TODO: generate presets menu if in gui mode
        from os.path import dirname, join
        Residue.set_templates_dir(join(dirname(__file__), "data"))

        session.change_tracker = ChangeTracker()
        session.pb_manager = PseudobondManager(session)

        from . import attr_registration
        session.attr_registration = attr_registration.RegAttrManager()
        session.custom_attr_preserver = attr_registration.CustomizedInstanceManager()

        session._atomic_command_handler = session.triggers.add_handler("command finished",
            lambda *args: check_for_changes(session))

        if session.ui.is_gui:
           session.ui.triggers.add_handler('ready', lambda *args, ses=session:
               _AtomicBundleAPI._add_presets_menu(ses))

    @staticmethod
    def finish(session, bundle_info):
        session.triggers.remove_handler(session._atomic_command_handler)

    @staticmethod
    def library_dir(bundle_info):
        from os.path import dirname, join
        return join(dirname(__file__), "lib")

    @staticmethod
    def _add_presets_menu(session):
        name_mapping = {
            'Stick': 'non-polymer',
            'Cartoon': 'small polymer',
            'Space-Filling (chain colors)': 'medium polymer',
            'Space-Filling (single color)': 'large polymer'
        }
        def callback(name, session=session):
            structures = [m for m in session.models if isinstance(m, Structure)]
            kw = {'set_lighting': len(structures) < 2}
            if name in name_mapping:
                kw['style'] = name_mapping[name]
            from .nucleotides.cmd import nucleotides
            nucleotides(session, 'atoms')
            for s in structures:
                atoms = s.atoms
                atoms.displays = True
                atoms.draw_modes = Atom.SPHERE_STYLE
                s.residues.ribbon_displays = False
                s.apply_auto_styling(**kw)
        for label in ['Original Look'] + sorted(list(name_mapping.keys())):
            session.ui.main_window.add_custom_menu_entry('Presets', label,
                lambda name=label: callback(name))

bundle_api = _AtomicBundleAPI()
