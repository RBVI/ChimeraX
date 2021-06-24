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
def combine_cmd(session, structures, *, close=False, model_id=None, name=None, new_chain_ids=True,
        ref_model=None):

    if structures is None:
        from chimerax.atomic import Structure
        structures = [m for m in session.models if isinstance(m, Structure)]
    if not structures:
        raise UserError("No structures specified")

    if ref_model is None:
        ref_model = structures[0]

    if model_id is None:
        model_id = session.models.next_id()
    #TODO

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
            ('new_chain_ids', BoolArg),
            ('ref_model', ModelArg)),
        ],
        synopsis = 'Copy/combine structure models')
    register('combine', cmd_desc, combine_cmd, logger=logger)
