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

from chimerax.core.state import StateManager
class _StructureAnisoManager(StateManager):
    def __init__(self, session, structure, *, from_session=False):
        self.init_state_manager(session, "structure thermal ellipsoids")
        self.session = session
        self.structure = structure
        from chimerax.atomic import Atoms
        self.shown_atoms = Atoms()
        self.atom_depictions = {}
        atoms = structure.atoms
        self._create_depictions(atoms[atoms.has_aniso_u])
        if not from_session:
            self._add_handlers()

    def destroy(self):
        for handler in self.handlers:
            handler.remove()
        self.structure = self.session = None
        super().destroy()

    def hide(self, atoms):
        """Hide thermal ellipsoids for these atoms"""
        #TODO
        pass

    def show(self, atoms):
        """Show thermal ellipsoids for these atoms"""
        cur_shown = len(self.shown_atoms)
        self.shown_atoms = self.shown_atoms.merge(atoms)
        if len(self.shown_atoms) == cur_shown:
            return
        #TODO: show (unhide) the depictions for atoms

    def _add_handlers(self):
        from chimerax.core.models import REMOVE_MODELS
        self.handlers = [
            #TODO
            #self.structure.triggers.add_handler('changes', self._changes_cb),
            #self.session.triggers.add_handler(REMOVE_MODELS, self._models_closed_cb)
        ]

    #TODO
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
            new_a = add_atom(old_a.name, old_a.element, dest_s.residues[0], coord, alt_loc=alt_loc)
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

def aniso_show(session, atoms=None):
    ''' Command to display thermal ellipsoids '''

    from chimerax.atomic import all_atoms
    if atoms is None:
        atoms = all_atoms(session)

    if not atoms:
        raise UserError("No atoms specified")

    atoms = atoms.filter(atoms.has_aniso_u)
    if not atoms:
        raise UserError("None of the specified atoms have anisotropic temperature factors")

    mgr_info = { mgr.structure: mgr for mgr in session.state_managers(_StructureAnisoManager) }

    for s, s_atoms in atoms.by_structure:
        if s not in mgr_info:
            mgr_info[s] = _StructureAnisoManager(session, s)
        mgr_info[s].show(atoms=s_atoms)

def aniso_hide(session, atoms=None):
    ''' Command to hide thermal ellipsoids '''

    from chimerax.atomic import all_atoms
    if atoms is None:
        atoms = all_atoms(session)

    if not atoms:
        raise UserError("No atoms specified")

    atoms = atoms.filter(atoms.has_aniso_u)
    if not atoms:
        raise UserError("None of the specified atoms have anisotropic temperature factors")

    mgr_info = { mgr.structure: mgr for mgr in session.state_managers(_StructureAnisoManager) }

    for s, s_atoms in atoms.by_structure:
        if s not in mgr_info:
            continue
        mgr_info[s].hide(atoms=s_atoms)

def register_command(logger):
    from chimerax.core.commands import register, CmdDesc
    from chimerax.core.commands import Or, EmptyArg
    from chimerax.atomic import AtomsArg

    desc = CmdDesc(required=[('atoms', Or(AtomsArg, EmptyArg))],
        #optional = [('locs', Or(ListOf(CharacterArg), EmptyArg)), ('residues', ResiduesArg)],
        synopsis='show thermal ellipsoids')
    register('aniso', desc, aniso_show, logger=logger)

    desc = CmdDesc(required=[('atoms', Or(AtomsArg, EmptyArg))], synopsis='hide thermal ellipsoids')
    register('~aniso', desc, aniso_hide, logger=logger)
