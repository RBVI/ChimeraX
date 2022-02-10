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

from chimerax.core.errors import UserError

def altlocs_change(session, alt_loc, residues=None):
    '''List altocs for residues

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
class _AltlocStateManager(StateManager):
    def __init__(self, session, base_residue):
        self.init_state_manager(session, "residue altlocs")
        self.session = session
        self.base_residue = base_residue
        self.alt_locs = { l: self._build_alt_loc(l) for l in base_residue.alt_locs } #TODO
        self.group = session.models.add_group(self.alt_locs.values(), name="%s alternate locations"
            % base_residue.string(omit_structure=True), parent=base_residue.structure)
        #TODO
        self.rotamers = list(rotamers) # don't want auto-shrinking of a Collection
        self.group = session.models.add_group(rotamers, name="%s rotamers"
            % base_residue.string(omit_structure=True), parent=base_residue.structure)
        from chimerax.atomic import get_triggers
        self.handler = get_triggers().add_handler('changes', self._changes_cb)
        from chimerax.core.triggerset import TriggerSet
        self.triggers = TriggerSet()
        self.triggers.add_trigger('fewer rotamers') # but not zero
        self.triggers.add_trigger('self destroyed')
        # below salvaged from swapaa interactive command...
        ret_val = []
        from . import swap_res
        from chimerax.atomic import AtomicStructures
        from chimerax.core.objects import Objects
        for r in residues:
            if res_type == "same":
                r_type = r.name
            else:
                r_type = res_type.upper()
            rotamers = swap_res.get_rotamers(session, r, res_type=r_type, rot_lib=rot_lib, log=log)
            mgr = _RotamerStateManager(session, r, rotamers)
            if session.ui.is_gui:
                from .tool import RotamerDialog
                RotamerDialog(session, "%s Side-Chain Rotamers" % r, mgr, res_type, rot_lib)
            ret_val.append(mgr)
            rot_structs = AtomicStructures(rotamers)
            rot_objects = Objects(atoms=rot_structs.atoms, bonds=rot_structs.bonds)
            from chimerax.std_commands.color import color
            color(session, rot_objects, color="byelement")
            from chimerax.std_commands.size import size
            size(session, rot_objects, stick_radius=0.1)

    def destroy(self):
        self.handler.remove()
        if self.group.id is not None:
            self.session.models.close([self.group])
        self.group = self.base_residue = self.rotamers = self.session = None
        super().destroy()

    def reset_state(self, session):
        self.triggers.activate_trigger('self destroyed', self)
        self.destroy()

    @classmethod
    def restore_snapshot(cls, session, data):
        return cls(session, data['base residue'], data['rotamers'])

    def take_snapshot(self, session, flags):
        data = {
            'base residue': self.base_residue,
            'rotamers': self.rotamers
        }
        return data

    def _changes_cb(self, trigger_name, changes):
        if changes.num_deleted_residues() == 0:
            return
        if self.base_residue.deleted:
            self.triggers.activate_trigger('self destroyed', self)
            self.destroy()
            return
        remaining = [rot for rot in self.rotamers if not rot.deleted]
        if len(remaining) < len(self.rotamers):
            if remaining:
                self.rotamers = remaining
                self.triggers.activate_trigger('fewer rotamers', self)
            else:
                self.triggers.activate_trigger('self destroyed', self)
                self.destroy()

def _get_alt_loc_residues(session, residues, locs):
    residues = [r for r in residues if r.alt_locs]
    if isinstance(locs, list):
        r_info = []
        locs = set(locs)
        rem_residues = []
        for r in residues:
            r_locs = set(r.alt_locs)
            not_found = locs - r_locs
            if not_found:
                from chimerax.core.command import commas
                session.logger.warning("%s does not have alt loc %s" % (r, commas(not_found)))
            found = locs & r_locs
            if found:
                r_info.append((r, found))
    else:
        r_info = [(r, r.alt_locs) for r in residues
    return r_info

def _gather_existing_mgrs(session, residues):
    mgr_info = {}
    residues = set(residues)
    for mgr in session.state_managers(_AltlocStateManager):
        if mgr.base_residue in residues:
            mgr_info[mgr.base_residue] = mgr
    return mgr_info

def altlocs_show(session, residues, *, locs=None):
    ''' Command to display non-current altlocs '''

    r_info = _get_alt_loc_residues(session, residues, locs)
    if not r_info:
        raise UserError("None of the specified residues have %salternate locations"
            % ("the specified " if isinstance(locs, list) else ""))

    mgr_info = _gather_existing_mgrs(session, [r for r, r_locs in r_info])

    # if locs is False, we can just nuke the existing managers
    if locs is False:
        for mgr in mgr_info.values():
            mgr.destroy()
        return []

    # create needed managers:
    for r, r_locs in r_info:
        if r in mgr_info:
            if locs is None:
                mgr_info[r].show()
            continue
        mgr_info[r] = _AltlocStateManager(session, r)

    if locs is not None:
        for mgr in mgr_info.values():
            mgr.show(locs)

    return list(mgr_info.values())

def register_command(logger):
    from chimerax.core.commands import register, CmdDesc, AnnotationError, StringArg, Or, CharacterArg
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
