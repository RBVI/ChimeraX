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
from chimerax.core.commands import commas, plural_form

def cmd_bond(session, *args, **kw):
    from .bond import create_bonds, CreateBondError
    try:
        created = create_bonds(*args, **kw)
    except CreateBondError as e:
        raise UserError(str(e))
    session.logger.info("Created %d %s" % (len(created), plural_form(created, "bond")))

def cmd_bond_length(session, bond, length=None, *, move="small"):
    if length is None:
        session.logger.info(("Bond length for %s is " + session.pb_dist_monitor.distance_format)
            % (bond, bond.length))
    else:
        from chimerax.atomic.struct_edit import set_bond_length
        set_bond_length(bond, length, move_smaller_side=(move=="small"))

def cmd_join_peptide(session, atoms, *, length=1.33, omega=180.0, phi=None, move="smaller"):
    # identify C-terminal carbon
    cs = atoms.filter(atoms.elements.names == "C")
    if not cs:
        raise UserError("No carbons in specified atoms")
    cs = cs.filter(cs.names == "C")
    if not cs:
        raise UserError("No carbons named 'C' in specified atoms")
    ctcs = []
    for c in cs:
        if not c.residue.chain or c.residue == c.residue.chain.existing_residues[-1]:
            ctcs.append(c)
    if not ctcs:
        raise UserError("No chain-terminal carbons in atoms")
    cts = []
    for ctc in ctcs:
        for nb in ctc.neighbors:
            if nb.name == "CA":
                cts.append(ctc)

    # identify N-terminal nitogen
    ns = atoms.filter(atoms.elements.names == "N")
    if not ns:
        raise UserError("No nitrogens in specified atoms")
    ns = ns.filter(ns.names == "N")
    if not ns:
        raise UserError("No nitrogens named 'N' in specified atoms")
    ntns = []
    for n in ns:
        if not n.residue.chain or n.residue == n.residue.chain.existing_residues[0]:
            ntns.append(n)
    if not ntns:
        raise UserError("No chain-initial nitrogens in atoms")
    nts = []
    for ntn in ntns:
        for nb in ntn.neighbors:
            if nb.name == "CA":
                nts.append(ntn)
    if not nts:
        raise UserError("Chain-initial %s %s not bonded to a CA atom"
            % (plural_form(ntns, 'nitrogen'), commas([str(ntn) for ntn in ntns])))

    if len(cts) > 1:
        if len(nts) > 1:
            raise UserError("Multiple N- and C-terminii in atoms")
        else:
            ntn = nts[0]
            ctcs = [ctc for ctc in cts if ctc.structure != ntn.structure]
            if len(ctcs) > 1:
                raise UserError("Multiple C-terminii in atoms")
            if len(ctcs) == 0:
                raise UserError("All C-termimii in atoms in same model as %s" % ntn)
            ctc = ctcs[0]
    else:
        ctc = ctcs[0]
        if len(nts) > 1:
            ntns = [ntn for ntn in nts if ntn.structure != ctc.structure]
            if len(ntns) > 1:
                raise UserError("Multiple N-terminii in atoms")
            if len(ntns) == 0:
                raise UserError("All N-termimii in atoms in same model as %s" % ctc)
        ntn = ntns[0]

    if move == "N":
        moving = ntn
    elif move == "C":
        moving = ctc
    else:
        smaller, larger = (ntn, ctc) if ctc.structure.num_atoms >= ntn.structure.num_atoms else (ctc, ntn)
        moving = smaller if move == "smaller" else larger

    from .mod import cn_peptide_bond
    return cn_peptide_bond(ctc, ntn, moving, length, omega, phi, log_chain_remapping=True)

def cmd_modify_atom(session, *args, **kw):
    from .mod import modify_atom, ParamError
    try:
        return modify_atom(*args, **kw)
    except ParamError as e:
        raise UserError(e)

def cmd_start_structure(session, method, model_info, subargs):
    from .manager import get_manager
    manager = get_manager(session)
    if manager.is_indirect(method):
        raise UserError("No command support for '%s' start-structure method" % method)
    if isinstance(model_info, str):
        from chimerax.atomic import AtomicStructure
        model = AtomicStructure(session, name=model_info, auto_style=manager.auto_style(method))
    else:
        model = model_info
    try:
        ret_val = manager.execute_command(method, model, subargs)
    except BaseException:
        if isinstance(model_info, str):
            model.delete()
        raise
    if model.num_atoms == 0:
        model.delete()
    elif isinstance(model_info, str):
        session.models.add([model])
    return ret_val

def register_command(command_name, logger):
    from chimerax.core.commands import CmdDesc, register, BoolArg, Or, IntArg, EnumOf, StringArg
    from chimerax.core.commands import DynamicEnum, RestOfLine, create_alias, PositiveFloatArg, FloatArg
    from chimerax.atomic import AtomArg, ElementArg, StructureArg, AtomsArg, BondArg
    from chimerax.atomic.bond_geom import geometry_name
    desc = CmdDesc(
        required=[('atoms', AtomsArg)],
        keyword = [('length', PositiveFloatArg),
            ('omega', FloatArg), ('phi', FloatArg),
            ('move', EnumOf(("large", "small", "N", "C")))],
        synopsis = 'join models through peptide bond'
    )
    register('build join peptide', desc, cmd_join_peptide, logger=logger)

    desc = CmdDesc(
        required=[('atom', AtomArg), ('element', ElementArg), ('num_bonds', IntArg)],
        keyword = [('geometry', EnumOf(range(len(geometry_name)), ids=geometry_name)),
            ('name', StringArg), ('connect_back', BoolArg), ('color_by_element', BoolArg),
            ('res_name', StringArg), ('new_res', BoolArg)],
        synopsis = 'modify atom'
    )
    register('build modify', desc, cmd_modify_atom, logger=logger)

    from .manager import get_manager
    manager = get_manager(logger.session)
    desc = CmdDesc(
        required=[('method', DynamicEnum(lambda mgr=manager: mgr.provider_names)),
            ('model_info', Or(StructureArg, StringArg)), ('subargs', RestOfLine)],
        keyword = [],
        synopsis = 'start structure'
    )
    register('build start', desc, cmd_start_structure, logger=logger)

    desc = CmdDesc(
        required=[('atoms', AtomsArg)],
        keyword = [('reasonable', BoolArg)],
        synopsis = 'add bond(s)'
    )
    register('bond', desc, cmd_bond, logger=logger)

    create_alias("~bond", "delete bonds $*", logger=logger)

    desc = CmdDesc(
        required=[('bond', BondArg)],
        optional = [('length', PositiveFloatArg)],
        keyword = [('move', EnumOf(("large", "small")))],
        synopsis = 'set bond length'
    )
    register('bond length', desc, cmd_bond_length, logger=logger)
