# vim: set expandtab ts=4 sw=4:

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

def log_chains(session, structures=None):
    if structures is None:
        from chimerax.atomic import AtomicStructure
        structures = [s for s in session.models if isinstance(s, AtomicStructure)]
    for s in structures:
        s._report_chain_descriptions(session)

def combine_cmd(session, structures, *, close=False, model_id=None, name=None):

    if structures is None:
        from chimerax.atomic import Structure
        structures = [m for m in session.models if isinstance(m, Structure)]
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
    #TODO custom attrs
    if close:
        session.models.close(structures)
        pass
    if model_id is not None:
        combination.id = model_id
    session.models.add([combination])
    return combination

def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, Or, EmptyArg, StringArg, BoolArg, ModelIdArg, \
        ModelArg, NoneArg
    from .args import StructuresArg, AtomicStructuresArg

    chains_desc = CmdDesc(
                        optional = [('structures', Or(AtomicStructuresArg, NoneArg))],
                        synopsis = 'Add structure chains table to the log'
    )
    register('log chains', chains_desc, log_chains, logger=logger)

    combine_desc = CmdDesc(
        required=[('structures', Or(StructuresArg,EmptyArg))],
        keyword=[
            ('close', BoolArg),
            ('model_id', ModelIdArg),
            ('name', StringArg),
            ('ref_model', ModelArg),
        ],
        synopsis = 'Copy/combine structure models')
    register('combine', combine_desc, combine_cmd, logger=logger)
