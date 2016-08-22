# vim: set expandtab ts=4 sw=4:

#
# Compute/assign secondary structure using Kabsch and Sander algorithm
#
def ss_assign(session, structs=None, *,
        min_helix_len=3, min_strand_len=3, energy_cutoff=-0.5, report=False):
    from chimerax.core.atomic import Structure
    if structs is None:
        structs = [m for m in session.models.list() if isinstance(m, Structure)]
    elif isinstance(structs, Structure):
        structs = [structs]

    if len(structs) == 0:
        from chimerax.core.errors import UserError
        raise UserError('No structures specified')

    from ._ksdssp import compute_ss
    for struct in structs:
        compute_ss(struct._c_pointer.value, energy_cutoff, min_helix_len, min_strand_len, report)

def register_command():
    from chimerax.core.commands import CmdDesc, register, StructuresArg, FloatArg, BoolArg, IntArg

    desc = CmdDesc(
        optional=[('structures', StructuresArg)],
        keyword=[('min_helix_len', IntArg),
                   ('min_strand_len', IntArg),
                   ('energy_cutoff', FloatArg),
                   ('report', BoolArg)],
    )
    register('ss_assign', desc, ss_assign)
