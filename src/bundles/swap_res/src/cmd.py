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

from chimerax.core.errors import UserError

default_criteria = "dchp"
def swap_aa(session, residues, res_type, *, angle_slop=None, bfactor=None, criteria=default_criteria,
    density=None, dist_slop=None, hbond_allowance=None, ignore_other_models=True, rot_lib=None, log=True,
    preserve=None, relax=True, retain=False, score_method="num", overlap_cutoff=None):
    ''' Command to swap amino acid side chains '''

    residues = _check_residues(residues)

    _check_num_residues(session, residues)

    if type(criteria) == str:
        for c in criteria:
            if c not in "dchp":
                raise UserError("Unknown criteria: '%s'" % c)
    elif preserve is not None:
        raise UserError("'preserve' not compatible with Nth-most-probable criteria")

    if rot_lib is None:
        rot_lib = session.rotamers.default_command_library_name

    if log:
        session.logger.info("Using %s library" % rot_lib)

    from . import swap_res
    swap_res.swap_aa(session, residues, res_type, bfactor=bfactor, clash_hbond_allowance=hbond_allowance,
        clash_score_method=score_method, clash_overlap_cutoff=overlap_cutoff,
        criteria=criteria, density=density, hbond_angle_slop=angle_slop,
        hbond_dist_slop=dist_slop, ignore_other_models=ignore_other_models, rot_lib=rot_lib, log=log,
        preserve=preserve, hbond_relax=relax, retain=retain)

def swap_na(session, residues, res_type, *, preserve=False, bfactor=None):
    # Can't swap P-only residues anyway, so just test for C4'
    if len(residues) == 0:
        raise UserError("No residues specified for swapping")
    import numpy
    nuc_residues = residues.filter(numpy.array([(pa is not None and pa.name == "C4'")
        for pa in residues.principal_atoms]))
    if len(nuc_residues) == 0:
        raise UserError("No nucleic acid residues specified for swapping")

    _check_num_residues(session, nuc_residues)

    from . import swap_res
    swap_res.swap_na(session, nuc_residues, res_type, bfactor=bfactor, preserve=preserve)

from chimerax.core.state import StateManager
class _RotamerStateManager(StateManager):
    def __init__(self, session, base_residue, rotamers, *, group=True):
        self.init_state_manager(session, "residue rotamers")
        self.session = session
        self.base_residue = base_residue
        self.rotamers = list(rotamers) # don't want auto-shrinking of a Collection
        # if 'group' is False, then an old session is restoring that didn't save the group info:
        # try to find the group in session.models
        # if 'group' is True, then create the group
        # Otherwise, 'group' is the group to use
        group_name = "%s rotamers" % base_residue.string(omit_structure=True)
        if group is True:
            self.group = session.models.add_group(rotamers, name=group_name, parent=base_residue.structure)
        elif group is False:
            for m in session.models:
                if m.name == group_name:
                    self.group = m
                    break
            else:
                raise AssertionError("Cannot find rotamer grouping model in open models")
        else:
            self.group = group
        from chimerax.atomic import get_triggers
        self.handler = get_triggers().add_handler('changes', self._changes_cb)
        from chimerax.core.triggerset import TriggerSet
        self.triggers = TriggerSet()
        self.triggers.add_trigger('fewer rotamers') # but not zero
        self.triggers.add_trigger('self destroyed')

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
        return cls(session, data['base residue'], data['rotamers'], group=data.get('group', False))

    def take_snapshot(self, session, flags):
        data = {
            'base residue': self.base_residue,
            'group': self.group,
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

def rotamers(session, residues, res_type, *, rot_lib=None, log=True):
    ''' Command to display possible side-chain rotamers '''

    residues = _check_residues(residues)

    if rot_lib is None:
        rot_lib = session.rotamers.default_command_library_name

    ret_val = []
    from . import swap_res
    from chimerax.atomic import AtomicStructures
    from chimerax.core.objects import Objects
    for r in residues:
        if res_type == "same":
            r_type = r.name
        else:
            r_type = res_type.upper()
        try:
            rotamers = swap_res.get_rotamers(session, r, res_type=r_type, rot_lib=rot_lib, log=log)
        except swap_res.NoResidueRotamersError:
            raise UserError("No rotamers for %s in %s rotamer library" % (r_type, rot_lib))
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
    return ret_val

def _check_residues(residues):
    residues = [r for r in residues if r.polymer_type == r.PT_AMINO]
    if not residues:
        raise UserError("No amino acid residues specified for swapping")
    return residues

def _check_num_residues(session, residues):
    if len(residues) > 2 and session.ui.is_gui and not session.in_script:
        from chimerax.ui.ask import ask
        if ask(session, "Really swap side chains for %d residues?" % len(residues),
                title="Confirm Swap") == "no":
            from chimerax.core.errors import CancelOperation
            raise CancelOperation("Swap %d side chains cancelled" % len(residues))

def register_command(command_name, logger):
    from chimerax.core.commands import CmdDesc, register, StringArg, BoolArg, NonNegativeIntArg, Or
    from chimerax.core.commands import NonNegativeFloatArg, DynamicEnum, ListOf, FloatArg, EnumOf
    from chimerax.atomic import ResiduesArg
    from chimerax.map import MapArg
    desc = CmdDesc(
        required = [('residues', ResiduesArg), ('res_type', StringArg)],
        keyword = [
            ('angle_slop', FloatArg),
            ('bfactor', FloatArg),
            ('criteria', Or(ListOf(NonNegativeIntArg), StringArg)),
            ('density', MapArg),
            ('dist_slop', FloatArg),
            ('hbond_allowance', FloatArg),
            ('ignore_other_models', BoolArg),
            ('rot_lib', DynamicEnum(logger.session.rotamers.library_names)),
            ('log', BoolArg),
            ('preserve', NonNegativeFloatArg),
            ('relax', BoolArg),
            ('retain', BoolArg),
            ('score_method', EnumOf(('sum', 'num'))),
            ('overlap_cutoff', FloatArg),
        ],
        synopsis = 'Swap amino acid side chain(s)'
    )
    register("swapaa", desc, swap_aa, logger=logger)

    desc = CmdDesc(
        required = [('residues', ResiduesArg), ('res_type', StringArg)],
        keyword = [
            ('rot_lib', DynamicEnum(logger.session.rotamers.library_names)),
            ('log', BoolArg),
        ],
        synopsis = 'Show possible side-chain rotamers'
    )
    register("swapaa interactive", desc, rotamers, logger=logger)

    desc = CmdDesc(
        required = [('residues', ResiduesArg), ('res_type', StringArg)],
        keyword = [
            ('bfactor', FloatArg),
            ('preserve', BoolArg),
        ],
        synopsis = 'Swap nucleic acid side chain(s)'
    )
    register("swapna", desc, swap_na, logger=logger)
