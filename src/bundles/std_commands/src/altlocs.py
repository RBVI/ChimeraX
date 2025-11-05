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

from chimerax.core.errors import UserError

def altlocs_change(session, alt_loc, residues=None):
    '''Change altocs for residues

    Parameters
    ----------
    alt_loc : single character
        alt_loc to change to
    residues : sequence or Collection of Residues
        change the altlocs for theses residues.  If not specified, then all residues.
    '''
    if residues is None:
        from chimerax.atomic import all_residues
        residues = all_residues(session)

    from chimerax.core.errors import UserError
    if not residues:
        UserError("No residues specified")

    num_changed = num_found = 0
    for r in residues:
        r_locs = set(r.atoms.alt_locs)
        r_locs.discard(' ')
        if len(r_locs) == 1 and alt_loc in r_locs:
            # already set to that alt loc
            num_found += 1
            continue
        try:
            r.set_alt_loc(alt_loc)
        except ValueError:
            pass
        else:
            num_changed += 1
            num_found += 1
    unchanged = num_found - num_changed

    from chimerax.core.commands import plural_form, commas
    if num_found == 0:
        session.logger.warning("Alternate location %s not found in %d %s" % (alt_loc, len(residues),
            plural_form(residues, "residue")))
    elif len(residues) == 1 and num_changed == 1:
        session.logger.info("Changed %s to alternate location %s" % (list(residues)[0], alt_loc))
    elif unchanged == 0:
        session.logger.info("Changed %d %s to alternate location %s" % (num_changed,
            plural_form(num_changed, "residue"), alt_loc))
    else:
        session.logger.info("Changed %d %s to alternate location %s (%d %s were already %s)" % (num_changed,
            plural_form(num_changed, "residue"), alt_loc, unchanged, plural_form(unchanged, "residue"),
            alt_loc))

def altlocs_clean(session, residues=None):
    '''Change current alt locs into non-alt locs and remove non-current alt locs

    Parameters
    ----------
    residues : sequence or Collection of Residues
        'Clean' the altlocs for theses residues.  If not specified, then all residues.
    '''
    if residues is None:
        from chimerax.atomic import all_residues
        residues = all_residues(session)

    from chimerax.core.errors import UserError
    if not residues:
        UserError("No residues specified")

    num_cleaned = 0
    for r in residues:
        # r.atoms.altlocs is the current alt locs
        r_locs = set([al for a in r.atoms for al in a.alt_locs])
        r_locs.discard(' ')
        if r_locs:
            r.clean_alt_locs()
            num_cleaned += 1

    from chimerax.core.commands import plural_form, commas
    if num_cleaned == 0:
        session.logger.info("No alternate locations in %d %s" % (len(residues),
            plural_form(len(residues), "residue")))
    else:
        session.logger.info("Removed alternate locations from %d %s" % (num_cleaned,
            plural_form(num_cleaned, "residue")))

def altlocs_list(session, residues=None):
    '''List altocs for residues

    Parameters
    ----------
    residues : sequence or Collection of Residues
        List the altlocs for theses residues.  If not specified, then all residues.
    '''
    if residues is None:
        from chimerax.atomic import all_residues
        residues = all_residues(session)

    from chimerax.core.errors import UserError
    if not residues:
        UserError("No residues specified")

    residues = sorted(residues)
    no_alt_locs = 0
    alt_locs = []
    for r in residues:
        # r.atoms.altlocs is the current alt locs
        r_locs = set([al for a in r.atoms for al in a.alt_locs])
        r_locs.discard(' ')
        if r_locs:
            alt_locs.append((r, r_locs))
        else:
            no_alt_locs += 1

    from chimerax.core.commands import plural_form, commas
    if no_alt_locs:
        session.logger.info("%d %s %s no alternate locations" % (no_alt_locs,
            plural_form(no_alt_locs, "residue"), plural_form(no_alt_locs, "has", plural="have")))

    for r, r_locs in alt_locs:
        used = set(r.atoms.alt_locs)
        used.discard(' ')
        session.logger.info("%s has alternate locations %s (using %s)" % (r,
            commas(sorted(r_locs), conjunction="and"), commas(used, conjunction="and")))

from chimerax.core.state import StateManager
class _StructureAltlocManager(StateManager):
    def __init__(self, session, structure, *, from_session=False):
        self.init_state_manager(session, "structure altlocs")
        self.session = session
        self.structure = structure
        self.expected_changes = set()
        if not from_session:
            from chimerax.core.models import Model
            self.main_group = Model("alternate locations", session)
            structure.add([self.main_group])
            self.res_alt_locs = {}
            self.res_group = {}
            for r in structure.residues:
                if not r.alt_locs:
                    continue
                self._build_alt_locs(r, self.main_group)
            self._add_handlers()

    def destroy(self):
        for handler in self.handlers:
            handler.remove()
        for al_to_s in self.res_alt_locs.values():
            for alt_loc_s in al_to_s.values():
                alt_loc_s._alt_loc_changes_handler.remove()
        if self.main_group.id is not None:
            self.session.models.close([self.main_group])
        self.group = self.structure = self.res_alt_locs = self.session = self.expected_changes = None
        super().destroy()

    def hide(self, residues=None, locs=None):
        if residues is None:
            residues = self.res_alt_locs.keys()
        for r in residues:
            if r not in self.res_alt_locs:
                continue
            if locs is None:
                locs = self.res_alt_locs[r].keys()
            for loc in locs:
                try:
                    al_model = self.res_alt_locs[r][loc]
                except KeyError:
                    continue
                al_model.display = False

    def show(self, residues=None, locs=None):
        if residues is None:
            residues = self.res_alt_locs.keys()
        for r in residues:
            if r not in self.res_alt_locs:
                continue
            if locs is None:
                locs = self.res_alt_locs[r].keys()
            for loc in locs:
                try:
                    al_model = self.res_alt_locs[r][loc]
                except KeyError:
                    continue
                al_model.display = True

    def _add_alt_loc_changes_handler(self, s, res, alt_loc):
        s._alt_loc_changes_handler = s.triggers.add_handler('changes',
            lambda trig_name, change_info, f=self._alt_loc_changes_cb, res=res, al=alt_loc:
            f(change_info, res, al))

    def _add_handlers(self):
        from chimerax.core.models import REMOVE_MODELS
        self.handlers = [
            self.structure.triggers.add_handler('changes', self._changes_cb),
            self.session.triggers.add_handler(REMOVE_MODELS, self._models_closed_cb)
        ]

    def _alt_loc_changes_cb(self, change_info, res, alt_loc):
        alt_loc_s, changes = change_info
        if alt_loc_s in self.expected_changes:
            # we made the changes ourself; leave them be
            self.expected_changes.remove(alt_loc_s)
            return
        for new_a in changes.created_atoms():
            alt_loc_s.delete_atom(new_a)

    def _build_alt_loc(self, res, alt_loc):
        from chimerax.atomic import AtomicStructure, Atom
        # _mirror_atoms code depends on structure name being the alt_loc
        s = AtomicStructure(self.session, name=alt_loc, auto_style=False, log_info=False)
        self.expected_changes.add(s)
        r = s.new_residue(res.name, res.chain_id, res.number, insert=res.insertion_code)
        self._update_alt_loc_res(alt_loc, res.atoms, s, {})
        s.display = False
        self.res_alt_locs.setdefault(res, {})[alt_loc] = s
        self._add_alt_loc_changes_handler(s, res, alt_loc)
        return s

    def _build_alt_locs(self, res, main_group):
        self.res_group[res] = self.session.models.add_group([self._build_alt_loc(res, al)
            for al in sorted(res.alt_locs)], name=res.string(omit_structure=True), parent=main_group)

    def _changes_cb(self, trigger_name, change_info):
        structure, changes = change_info
        if structure.deleted:
            self.destroy()
            return
        del_groups = []
        del_alt_locs = []
        if changes.num_deleted_residues() > 0:
            for r, group in self.res_group.items():
                if r.deleted:
                    del_groups.append(group)
        if 'alt_locs changed' in changes.residue_reasons():
            for r, group in self.res_group.items():
                if r.deleted:
                    continue
                r_alt_locs = r.alt_locs
                if not r_alt_locs:
                    del_groups.append(group)
                else:
                    shown_alt_locs = set(self.res_alt_locs[r].keys())
                    for del_al in shown_alt_locs - r_alt_locs:
                        del_alt_locs.append(self.res_alt_locs[r][del_al])

        if del_groups:
            if len(del_groups) == len(self.res_group):
                self.destroy()
                return
            # self.res_group will be cleaned up in the models-closed callback
            self.session.models.close(del_groups)
        if del_alt_locs:
            self.session.models.close(del_alt_locs)
        mod_res = set()
        for a in changes.created_atoms():
            if a.residue in self.res_group:
                mod_res.add(a.residue)
        if mod_res:
            # defer the mirroring to the altloc model in order to allow atoms added directly to the model
            # (e.g. from addh) to first be removed
            from chimerax.atomic import get_triggers
            get_triggers().add_handler('changes done',
                lambda *args, f=self._mirror_atoms, residues=mod_res: f(residues))
        for r in changes.created_residues():
            if r.alt_locs:
                self._build_alt_locs(r, self.main_group)

    def _mirror_atoms(self, residues):
        from chimerax.atomic import AtomicStructure
        from chimerax.atomic.struct_edit import add_atom
        for r in residues:
            group = self.res_group[r]
            for alt_loc_s in group.child_models():
                if not isinstance(alt_loc_s, AtomicStructure) or len(alt_loc_s.name) != 1:
                    continue
                self.expected_changes.add(alt_loc_s)
                alt_loc = alt_loc_s.name
                name_lookup = { a.name: a for a in alt_loc_s.atoms }
                # find newly created atoms, and correspondences between residue atoms and altloc model atoms
                atom_map = {}
                created = []
                for a in r.atoms:
                    try:
                        atom_map[a] = name_lookup[a.name]
                    except KeyError:
                        created.append(a)
                assert(len(atom_map) == alt_loc_s.num_atoms)
                self._update_alt_loc_res(alt_loc, created, alt_loc_s, atom_map)
        from chimerax.core.triggerset import DEREGISTER
        return DEREGISTER

    def _models_closed_cb(self, trigger_name, closed_models):
        if self.structure in closed_models or self.main_group in closed_models:
            self.destroy()
            return

        if not self.main_group.child_models():
            self.destroy()
            return

        # check the altloc models:
        closures = []
        for r, alt_locs in list(self.res_alt_locs.items()):
            for alt_loc, al_s in list(alt_locs.items()):
                if al_s in closed_models:
                    al_s._alt_loc_changes_handler.remove()
                    del alt_locs[alt_loc]
            if not alt_locs:
                del self.res_alt_locs[r]
                res_group = self.res_group[r]
                del self.res_group[r]
                if res_group not in closed_models:
                    closures.append(res_group)
        if closures:
            self.session.models.close(closures)

    def _update_alt_loc_res(self, alt_loc, source_atoms, dest_s, atom_map):
        from chimerax.atomic.struct_edit import add_atom
        from chimerax.atomic import Atom, Atoms, Bonds
        new_atoms = []
        for old_a in source_atoms:
            use_alt_loc = alt_loc in old_a.alt_locs
            coord = old_a.get_alt_loc_coord(alt_loc) if use_alt_loc else old_a.coord
            new_a = add_atom(old_a.name, old_a.element, dest_s.residues[0], coord, alt_loc=alt_loc,
                occupancy=old_a.get_alt_loc_occupancy(alt_loc) if use_alt_loc else old_a.occupancy)
            new_atoms.append(new_a)
            new_a.draw_mode = Atom.STICK_STYLE
            atom_map[old_a] = new_a
            if not old_a.is_side_chain and not use_alt_loc:
                new_a.display = False
        new_bonds = []
        for old_a in source_atoms:
            for old_b in old_a.bonds:
                a1, a2 = old_b.atoms
                try:
                    new1 = atom_map[a1]
                    new2 = atom_map[a2]
                except KeyError:
                    continue
                if new2 not in new1.neighbors:
                    new_bonds.append(dest_s.new_bond(new1, new2))
        from chimerax.core.objects import Objects
        alt_loc_objects = Objects(atoms=Atoms(new_atoms), bonds=Bonds(new_bonds))
        from chimerax.std_commands.color import color
        color(self.session, alt_loc_objects, color="byelement")
        from chimerax.std_commands.size import size
        size(self.session, alt_loc_objects, stick_radius=0.1, verbose=False)

    def reset_state(self, session):
        self.destroy()

    @classmethod
    def restore_snapshot(cls, session, data):
        inst = cls(session, data['structure'], from_session=True)
        inst.main_group = data['main_group']
        inst.res_alt_locs = data['res_alt_locs']
        inst.res_group = data['res_group']
        def delayed_registration(*args, inst=inst):
            inst._add_handlers()
            for r, lookup in inst.res_alt_locs.items():
                for alt_loc, al_model in lookup.items():
                    inst._add_alt_loc_changes_handler(al_model, r, alt_loc)
            from chimerax.core.triggerset import DEREGISTER
            return DEREGISTER
        from chimerax.atomic import get_triggers
        get_triggers().add_handler('changes done', delayed_registration)
        return inst

    def take_snapshot(self, session, flags):
        data = {
            'structure': self.structure,
            'main_group': self.main_group,
            'res_alt_locs': self.res_alt_locs,
            'res_group': self.res_group,
        }
        return data

def altlocs_show(session, locs=None, residues=None):
    ''' Command to display non-current altlocs '''

    from chimerax.atomic import all_residues, Residues
    if residues is None:
        residues = all_residues(session)

    res_locs = { r: set(r.alt_locs) for r in residues if r.alt_locs }

    if not res_locs:
        raise UserError("None of the specified residues have alternate locations")

    if locs:
        test_locs = set(locs)
        residues = [r for r, r_locs in res_locs.items() if test_locs & r_locs]
        if not residues:
            raise UserError("None of the specified residues have the requested alternate locations")
    else:
        residues = list(res_locs.keys())
    residues = Residues(residues)

    mgr_info = { mgr.structure: mgr for mgr in session.state_managers(_StructureAltlocManager) }

    for s, s_residues in residues.by_structure:
        if s not in mgr_info:
            mgr_info[s] = _StructureAltlocManager(session, s)
        mgr_info[s].show(residues=s_residues, locs=locs)

def altlocs_hide(session, locs=None, residues=None):
    ''' Command to hide non-current altlocs '''

    from chimerax.atomic import all_residues, Residues
    if residues is None:
        residues = all_residues(session)

    res_locs = { r: set(r.alt_locs) for r in residues if r.alt_locs }

    if not res_locs:
        raise UserError("None of the specified residues have alternate locations")

    if locs:
        test_locs = set(locs)
        residues = [r for r, r_locs in res_locs.items() if test_locs & r_locs]
        if not residues:
            raise UserError("None of the specified residues have the requested alternate locations")
    else:
        residues = list(res_locs.keys())
    residues = Residues(residues)

    mgr_info = { mgr.structure: mgr for mgr in session.state_managers(_StructureAltlocManager) }

    for s, s_residues in residues.by_structure:
        if s not in mgr_info:
            continue
        mgr_info[s].hide(residues=s_residues, locs=locs)

def register_command(logger):
    from chimerax.core.commands import register, CmdDesc, AnnotationError, StringArg, CharacterArg, ListOf
    from chimerax.core.commands import Or, EmptyArg
    from chimerax.atomic import ResiduesArg

    desc = CmdDesc(required=[('alt_loc', CharacterArg)],
        optional = [('residues', ResiduesArg)],
        synopsis='change alternate atom locations')
    register('altlocs change', desc, altlocs_change, logger=logger)

    desc = CmdDesc(optional = [('residues', ResiduesArg)],
        synopsis='change current alternate atom locations into non-alt-locs and remove other alt locs')
    register('altlocs clean', desc, altlocs_clean, logger=logger)

    desc = CmdDesc(optional = [('residues', ResiduesArg)],
        synopsis='list alternate atom locations')
    register('altlocs list', desc, altlocs_list, logger=logger)

    desc = CmdDesc(optional = [('locs', Or(ListOf(CharacterArg), EmptyArg)), ('residues', ResiduesArg)],
        synopsis='show alternate atom locations')
    register('altlocs show', desc, altlocs_show, logger=logger)

    desc = CmdDesc(optional = [('locs', Or(ListOf(CharacterArg), EmptyArg)), ('residues', ResiduesArg)],
        synopsis='hide alternate atom locations')
    register('altlocs hide', desc, altlocs_hide, logger=logger)
