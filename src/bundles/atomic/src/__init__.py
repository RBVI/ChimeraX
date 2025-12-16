# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This particular file is part of the ChimeraX library. You can also
# redistribute and/or modify it under the terms of the GNU Lesser General
# Public License version 2.1 as published by the Free Software Foundation.
# For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
# EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
# LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
# VERSION 2.1
#
# This notice must be embedded in or attached to all copies, including partial
# copies, of the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===
import os

# Atomic sticks its library in the same directory as other files
def get_lib() -> str:
    return os.path.dirname(__file__)

# ensure atomic_libs C++ shared libs are linkable by us
import chimerax.atomic_lib

from .molobject import Atom, Bond, Chain, CoordSet, Element, Pseudobond, Residue, Sequence, \
    StructureSeq, PseudobondManager, Ring, ChangeTracker, StructureData
from .molobject import SeqMatchMap, estimate_assoc_params, try_assoc, StructAssocError
from .molobject import next_chain_id, chain_id_characters
# pbgroup must precede molarray since molarray uses interatom_pseudobonds in global scope
from .pbgroup import PseudobondGroup, all_pseudobond_groups, all_pseudobonds
from .pbgroup import interatom_pseudobonds, selected_pseudobonds
from .molarray import Collection, Atoms, AtomicStructures, Bonds, Chains, CoordSets, Pseudobonds, \
    Structures, PseudobondGroups, Residues, concatenate
from .structure import AtomicStructure, Structure, LevelOfDetail
from .structure import selected_atoms, selected_bonds, selected_residues, selected_chains
from .structure import all_atoms, all_bonds, all_residues, all_atomic_structures, all_structures
from .structure import structure_atoms, structure_residues, structure_graphics_updater, level_of_detail
from .structure import PickedAtom, PickedBond, PickedResidue, PickedPseudobond
from .structure import uniprot_ids
from .molsurf import buried_area, MolecularSurface, surfaces_with_atoms
from .changes import check_for_changes
from .pdbmatrices import biological_unit_matrices
from .triggers import get_triggers
from .shapedrawing import AtomicShapeDrawing, AtomicShapeInfo
from .args import ElementArg, AtomArg, AtomsArg, OrderedAtomsArg, ResiduesArg
from .args import BondArg, BondsArg, PseudobondsArg, PseudobondGroupsArg
from .args import UniqueChainsArg, ChainArg, SequencesArg, SequenceArg, UniProtIdArg, is_uniprot_id
from .args import AtomicStructureArg, AtomicStructuresArg, StructureArg, StructuresArg
from .args import SymmetryArg, concise_residue_spec, concise_chain_spec
from .cytmpl import TmplResidue

def initialize_atomic(session):
    from . import settings
    settings.settings = settings._AtomicSettings(session, "atomic")

    from chimerax import atomic_lib, pdb_lib # ensure libs we need are linked
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

    # for efficiency when destroying many structures, batch the updating of Collections
    from chimerax.core.models import BEGIN_DELETE_MODELS, END_DELETE_MODELS
    session.triggers.add_handler(BEGIN_DELETE_MODELS, Structure.begin_destructor_batching)
    session.triggers.add_handler(END_DELETE_MODELS, Structure.end_destructor_batching)

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
            from chimerax.core.session import State, StateManager
            base_class = StateManager if class_name.endswith("Manager") else State
            class Fake(base_class):
                def clear(self):
                    pass
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
        if mgr == session.presets:
            from .presets import run_preset
            run_preset(session, name, mgr, **kw)
        elif mgr.name == "items inspection":
            from .inspectors import item_options
            return item_options(session, name, **kw)
        else:
            class_obj = {'atoms': Atom, 'residues': Residue, 'chains': Chain, 'structures': Structure }[name]
            from chimerax.render_by_attr import RenderAttrInfo
            class Info(RenderAttrInfo):
                _class_obj = class_obj
                monitorings = set()
                handler = None

                def attr_change_notify(self, attr_name, callback, *, prefix=name[:-1]):
                    if callback is None:
                        self.monitorings.discard(attr_name)
                        if not self.monitorings and self.handler:
                            self.handler.remove()
                            self.handler = None
                    else:
                        self.monitorings.add(attr_name)
                        if not self.handler:
                            def handler(trig_name, changes, *, attr_name=attr_name, prefix=prefix,
                                    callback=callback):
                                if attr_name + " changed" in getattr(changes, prefix + "_reasons")():
                                    callback()
                            from .triggers import get_triggers
                            self.handler = get_triggers().add_handler('changes', handler)

                @property
                def class_object(self):
                    return self._class_obj

                def deworm_applicable(self, models):
                    for m in models:
                        if getattr(m, 'worm_ribbon', False):
                            return True
                    return False

                def hide_attr(self, attr_name, rendering):
                    if rendering:
                        if self.class_object == Chain and attr_name.startswith('num') \
                        and attr_name.endswith('residues'):
                            return False
                        if self.class_object == Chain and attr_name == 'polymer_type':
                            return True
                    else:
                        if self.class_object == Atom and attr_name in ['is_side_connector', 'num_bonds',
                                'num_explicit_bonds', 'selected', 'visible']:
                            return True
                    return super().hide_attr(attr_name, rendering)

                def model_filter(self, model):
                    return isinstance(model, Structure)

                def render(self, session, attr_name, models, method, params, sel_only):
                    prefix = { Atom: 'a', Residue: 'r', Chain: 'c', Structure: 'm' }[self.class_object]
                    from chimerax.core.commands import run, concise_model_spec, StringArg
                    spec = concise_model_spec(session, models)
                    if sel_only:
                        if not session.selection.empty():
                            if spec:
                                spec += " & sel"
                            else:
                                spec = "sel"
                    if method == "color":
                        targets, spectrum = params
                        letters = ""
                        for target in targets:
                            if target == "atoms":
                                letters += "ab"
                            elif target == "cartoons":
                                letters += "c"
                            elif target == "surfaces":
                                letters += "s"
                        from chimerax.core.colors import color_name
                        no_val_string = ""
                        palette_vals = []
                        for val, rgba in spectrum:
                            cname = color_name([int(v*255 + 0.5) for v in rgba])
                            if val is None:
                                no_val_string = " noValueColor %s" % StringArg.unparse(cname)
                            else:
                                palette_vals.append((val,cname))
                        if palette_vals:
                            if len(palette_vals) == 1:
                                palette_vals.append(palette_vals[0])
                            palette_string = "palette %s" % StringArg.unparse(":".join(["%g,%s" % (v,c)
                                for v, c in palette_vals]))
                        else:
                            palette_string = ""
                        run(session, "color byattr %s:%s %s target %s %s%s" % (prefix, attr_name, spec,
                            letters, palette_string, no_val_string))
                    elif method == "radius":
                        atom_style, way_points = params
                        wp_string = self._way_point_string(way_points)
                        from chimerax.std_commands.size import AtomRadiiStyleArg
                        if atom_style == AtomRadiiStyleArg.default:
                            style_arg = ""
                        else:
                            style_arg = " style %s" % atom_style
                        run(session, "size byattr %s:%s %s%s%s" % (prefix, attr_name, spec, wp_string,
                            style_arg))
                    elif method == "worm":
                        show_worms, way_points = params
                        if show_worms:
                            wp_string = self._way_point_string(way_points)
                            run(session, "cartoon byattr %s:%s %s%s" % (prefix, attr_name, spec, wp_string))
                        else:
                            run(session, "~worm %s" % spec)

                def select(self, session, attr_name, models, discrete, params):
                    prefix = { Atom: '@@', Residue: '::', Chain: '//', Structure: '##' }[self.class_object]
                    from chimerax.core.commands import run, concise_model_spec, StringArg, BoolArg, FloatArg
                    spec = concise_model_spec(session, models)
                    if spec and self.class_object == Structure:
                        spec += ' & '
                    if discrete:
                        if None in params:
                            params.remove(None)
                            spec += f"{prefix}^{attr_name}"
                        for attr_val in params:
                            arg = BoolArg if isinstance(attr_val, bool) else StringArg
                            spec += f"{prefix}{attr_name}={arg.unparse(attr_val)}"
                    else:
                        if params is None:
                            spec += f'{prefix}^{attr_name}'
                        else:
                            between, low, high = params
                            if between:
                                spec += f'{prefix}{attr_name}>={FloatArg.unparse(low)} & ' \
                                    f'{prefix}{attr_name}<={FloatArg.unparse(high)}'
                            else:
                                spec += f'{prefix}{attr_name}<{FloatArg.unparse(low)} | ' \
                                    f'{prefix}{attr_name}>{FloatArg.unparse(high)}'
                    run(session, "select " + spec)

                def values(self, attr_name, models):
                    if self._class_obj == Atom:
                        collections = [m.atoms for m in models]
                    elif self._class_obj == Residue:
                        collections = [m.residues for m in models]
                    elif self._class_obj == Chain:
                        collections = [m.chains for m in models]
                    else:
                        collections = [Structures(models)]
                    from chimerax.core.commands import plural_of
                    collections = concatenate(collections)
                    plural_attr = plural_of(attr_name)
                    try:
                        all_vals = getattr(concatenate(collections), plural_of(attr_name))
                    except AttributeError:
                        all_vals = [getattr(item, attr_name, None) for item in collections]
                    import numpy
                    if not isinstance(all_vals, numpy.ndarray):
                        all_vals = numpy.array(all_vals)
                    non_none_vals = all_vals[all_vals != None]
                    return non_none_vals, len(non_none_vals) < len(all_vals)

                def _way_point_string(self, way_points):
                    no_val_string = ""
                    wp_vals = []
                    for attr_val, radius in way_points:
                        if attr_val is None:
                            no_val_string = " noValueRadius %g" % radius
                        else:
                            wp_vals.append((attr_val, radius))
                    if wp_vals:
                        wp_string = ' ' + " ".join(["%g:%g" % (av,rad) for av, rad in wp_vals])
                    else:
                        wp_string = ""
                    return "%s%s"% (wp_string, no_val_string)
            return Info(session)

    @staticmethod
    def finish(session, bundle_info):
        session.triggers.remove_handler(session._atomic_command_handler)

    @staticmethod
    def register_selector(selector_name, logger):
        # 'register_selector' is lazily called when selector is referenced
        from .selectors import register_selectors
        register_selectors(logger)

    @staticmethod
    def register_command(command_name, logger):
        # 'register_command' is lazily called when command is referenced
        from . import cmd
        cmd.register_command(logger)

    @staticmethod
    def _add_gui_items(session):
        from .selectors import add_select_menu_items
        add_select_menu_items(session)

        from .contextmenu import add_selection_context_menu_items
        add_selection_context_menu_items(session)
        
bundle_api = _AtomicBundleAPI()
