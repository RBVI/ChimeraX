# vim: set expandtab ts=4 sw=4:

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

def log_chains(session, structures=None):
    if structures is None:
        from chimerax.atomic import AtomicStructure
        structures = [s for s in session.models if isinstance(s, AtomicStructure)]
    for s in structures:
        s._report_chain_descriptions(session)

def combine_cmd(session, structures, *, close=False, model_id=None, name=None):

    if structures is None:
        from chimerax.atomic import AtomicStructure
        structures = [m for m in session.models if isinstance(m, AtomicStructure)]
    else:
        structures = list(structures)
    if not structures:
        raise UserError("No structures specified")

    if name is None:
        if len(structures) == 1:
            name = "copy of " + structures[0].name
        else:
            name = "combination"

    combination = structures[0].copy(name)

    # Compute needed remapping of chain IDs
    seen_ids = set(combination.residues.unique_chain_ids)
    for s in structures[1:]:
        chain_id_mapping = {}
        chain_ids = sorted(s.residues.unique_chain_ids)
        for chain_id in chain_ids:
            if chain_id in seen_ids:
                from chimerax.atomic import next_chain_id
                new_id = next_chain_id(chain_id)
                while new_id in seen_ids or new_id in chain_ids:
                    new_id = next_chain_id(new_id)
                session.logger.info("Remapping chain ID '%s' in %s to '%s'" % (chain_id, s, new_id))
                chain_id_mapping[chain_id] = new_id
                seen_ids.add(new_id)
            else:
                seen_ids.add(chain_id)
        combination.combine(s, chain_id_mapping, structures[0].scene_position)
    combination.position = structures[0].scene_position
    if close:
        # also close empty parent models
        closures = set(structures)
        parents = set([m.parent for m in closures if m.parent is not None])
        while True:
            new_parents = set()
            for parent in parents:
                children = set(parent.child_models())
                if children <= closures:
                    if parent.parent is not None: # don't try to close root model!
                        new_parents.add(parent.parent)
                        closures -= children
                        closures.add(parent)
            if new_parents:
                parents = new_parents
                new_parents.clear()
            else:
                break
        session.models.close(list(closures))
    if model_id is not None:
        combination.id = model_id
    session.models.add([combination])
    return combination

def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, Or, EmptyArg, StringArg, BoolArg, ModelIdArg, \
        NoneArg
    from .args import AtomicStructuresArg

    chains_desc = CmdDesc(
                        optional = [('structures', Or(AtomicStructuresArg, NoneArg))],
                        synopsis = 'Add structure chains table to the log'
    )
    register('log chains', chains_desc, log_chains, logger=logger)

    combine_desc = CmdDesc(
        required=[('structures', Or(AtomicStructuresArg,EmptyArg))],
        keyword=[
            ('close', BoolArg),
            ('model_id', ModelIdArg),
            ('name', StringArg),
        ],
        synopsis = 'Copy/combine structure models')
    register('combine', combine_desc, combine_cmd, logger=logger)
