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
def combine_cmd(session, structures, *, close=False, model_id=None, name=None, ref_model=None):

    if structures is None:
        from chimerax.atomic import Structure
        structures = [m for m in session.models if isinstance(m, Structure)]
    if not structures:
        raise UserError("No structures specified")

    # sort structures by size, so that the faster copying can be used on the largest structure
    structures = sorted(structures, key=lambda s: 0 - s.num_atoms)
    if ref_model is None:
        ref_model = structures[0]

    if model_id is None:
        model_id = session.models.next_id()

    if name is None:
        if len(structures) == 1:
            name = "copy of " + ref_model.name
        else:
            name = "combination"

    # Compute needed remapping of chain IDs
    seen_ids = set()
    chain_id_mapping = {}
    for s in structures:
        chain_ids = sorted(s.residues.unique_chain_ids)
        for chain_id in chain_ids:
            if chain_id in seen_ids:
                from chimerax.atomic import next_chain_id
                new_id = next_chain_id(chain_id)
                while new_id in seen or new_id in chain_ids:
                    new_id = next_chain_id(new_id)
                session.logger.info("Remapping chain ID '%s' in %s to '%s'" % (chain_id, s, new_id))
                chain_id_mapping[(s, chain_id)] = new_id
                seen_ids.add(new_id)
            else:
                seen_ids.add(chain_id)

    combination = structures[0].copy(name)
    #TODO: adjust to ref_model
    for s in structures[1:]:
        #TODO
    session.models.add([combination])
    return combination

def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, Or, EmptyArg, StringArg, BoolArg, ModeIdArg, \
        ModelArg
    from chimerax.atomic import StructuresArg
    cmd_desc = CmdDesc(
        required=[('structures', Or(StructuresArg,EmptyArg))],
        keyword=[
            ('close', BoolArg),
            ('model_id', ModeIdArg),
            ('name', StringArg),
            ('ref_model', ModelArg)),
        ],
        synopsis = 'Copy/combine structure models')
    register('combine', cmd_desc, combine_cmd, logger=logger)
