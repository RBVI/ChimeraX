# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
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
        from chimerax.core.undo import UndoState
        from chimerax.atomic.struct_edit import set_bond_length
        undo_state = UndoState("bond length")
        set_bond_length(bond, length, move_smaller_side=(move=="small"), undo_state=undo_state)
        session.undo.register(undo_state)

def cmd_invert(session, atoms):
    if len(atoms) not in [1,2]:
        raise UserError("Must specify exactly 1 or 2 atoms; you specified %d" % len(atoms))
    if len(atoms) == 1:
        center = atoms[0]
        kw = {}
    else:
        a1, a2 = atoms
        common_neighbors = set(a1.neighbors) & set (a2.neighbors)
        if len(common_neighbors) == 1:
            center = common_neighbors.pop()
            kw = { 'swapees': atoms }
        elif common_neighbors:
            raise UserError("%s and %s have more than one neighbor in common!" % (a1, a2))
        else:
            raise UserError("%s and %s have no neighbor atoms in common!" % (a1, a2))
    from .mod import invert_chirality, InvertChiralityError
    try:
        invert_chirality(center, **kw)
    except InvertChiralityError as e:
        raise UserError(str(e))

def cmd_join_bond(session, atoms, *, length=None, move="small", dihedral=None, dihedral_atoms=None):
    from .mod import bind, check_join_bond_atoms
    okay, message = check_join_bond_atoms(atoms)
    if not okay:
        raise UserError(message)

    if length is None:
        from chimerax.atomic import Element
        length = Element.bond_length(atoms[0].neighbors[0].element, atoms[1].neighbors[0].element)

    if move == "small":
        move = atoms[0].structure \
            if atoms[0].structure.num_atoms < atoms[1].structure.num_atoms else atoms[1].structure
    elif move == "large":
        move = atoms[0].structure \
            if atoms[0].structure.num_atoms > atoms[1].structure.num_atoms else atoms[1].structure
    else:
        if move not in [atoms[0].structure, atoms[1].structure]:
            raise UserError("'move' structure not one of the structures being bonded!")

    if dihedral is None:
        if dihedral_atoms is not None:
            raise UserError("Specified 'dihedralAtoms' but not 'dihedral'!")
    else:
        if dihedral_atoms is None:
            dihedral_atoms = []
            for a in atoms:
                neighbor = a.neighbors[0]
                if neighbor.num_bonds == 1:
                    raise UserError(
                        "%s has no other bonded atoms, so do not specify 'dihedral' argument" % a)
                if neighbor.num_bonds > 2:
                    raise UserError(
                        "%s has multiple other bonded atoms, so must specify 'dihedralAtoms' argument" % a)
                for nnb in neighbor.neighbors:
                    if nnb != a:
                        dihedral_atoms.append(nnb)
                        break
        else:
            if len(dihedral_atoms) != 2:
                raise UserError("Must specify exactly two dihedral atoms; you specified %d"
                    % len(dihedral_atoms))
            dihed_msg = "The dihedral atoms must be already bonded to the newly bonded atoms" \
                " (one on each end)"
            if dihedral_atoms[0].structure == dihedral_atoms[1].structure:
                raise UserError(dihed_msg)
            # get the dihdral atoms in the same order as atoms
            candidates = []
            for a in atoms:
                for nb in a.neighbors:
                    for nnb in nb.neighbors:
                        if nnb == a:
                            continue
                        if nnb in dihedral_atoms:
                            candidates.append(nnb)
                            break
                    else:
                        raise UserError("'dihedralAtoms' does not incude any of the atoms bonded to %s" % nb)
            dihedral_atoms = candidates

    # second atom arg to bind() gets moved, so...
    if move == atoms[1].structure:
        a1, a2 = atoms
        if dihedral_atoms:
            d1, d2 = dihedral_atoms
    else:
        a2, a1 = atoms
        if dihedral_atoms:
            d2, d1 = dihedral_atoms
    if dihedral is None:
        dihed_info = None
    else:
        dihed_info = [((d1, a1, a2, d2), dihedral)]
    return bind(a1, a2, length, dihed_info, renumber=move, log_chain_remapping=True)

def cmd_join_peptide(session, atoms, *, length=1.33, omega=180.0, phi=None, move="small"):
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
        moving = smaller if move == "small" else larger

    from .mod import cn_peptide_bond
    return cn_peptide_bond(ctc, ntn, moving, length, omega, phi, log_chain_remapping=True)

def cmd_modify_atom(session, *args, **kw):
    from .mod import modify_atom, ParamError
    try:
        return modify_atom(*args, **kw)
    except ParamError as e:
        raise UserError(e)

def _check_connected_polymeric(residues):
    chain = None
    for r in residues:
        if r.chain is None:
            return False
        if chain is None:
            chain = r.chain
            last_index = r.chain.residues.index(r)
            continue
        if chain != r.chain:
            return False
        last_index += 1
        while last_index < len(chain):
            next_r = chain.residues[last_index]
            if next_r is None:
                last_index += 1
                continue
            if next_r != r:
                return False
            break
        else:
            return False
    return chain is not None

def cmd_replace_residues(session, replaced_residues, with_=None, *,
        bond_start=None, bond_end=None, numbering_start=None):
    # If bond_start/end is None then bond the start/end of the replacement to adjacent residue if
    # bond length seems reasonable.  If numbering_start is None, use the first replaced residue's
    # number as the start
    for checked, msg_frag in [(replaced_residues, "Target"), (with_, "Replacement")]:
        if not _check_connected_polymeric(checked):
            raise UserError("%s residues are not directly chain-connected polymeric residues" % msg_frag)
    reset_chain_info = [rr.name for rr in replaced_residues] != [wr.name for wr in with_]
    before = after = None
    replaced = set(replaced_residues)
    replaced_seen = False
    for r in replaced_residues[0].chain.residues:
        if not r:
            continue
        if r in replaced:
            replaced_seen = True
            continue
        if replaced_seen:
            after = r
            break
        else:
            before = r
    if numbering_start is None:
        number = replaced_residues[0].number
    else:
        number = numbering_start
    chain = replaced_residues[0].chain
    s = chain.structure
    for r in replaced_residues:
        s.delete_residue(r)

    from chimerax.atomic.struct_edit import add_atom, add_bond
    atom_mapping = {}
    used_numbers = set([r.number for r in s.residues if r.chain_id == chain.chain_id])
    transform = s.scene_position.inverse()
    replacements = []
    for from_r in with_:
        while number in used_numbers:
            number += 1
        used_numbers.add(number)
        replacement_r = s.new_residue(from_r.name, chain.chain_id, number, precedes=after)
        replacements.append(replacement_r)
        for from_a in from_r.atoms:
            pos = transform * from_a.scene_coord
            replace_a = add_atom(from_a.name, from_a.element, replacement_r, pos, info_from=from_a)
            replace_a.color = from_a.color
            atom_mapping[from_a] = replace_a
            for from_nb, from_bond in zip(from_a.neighbors, from_a.bonds):
                try:
                    replace_nb = atom_mapping[from_nb]
                except KeyError:
                    continue
                add_bond(replace_a, replace_nb, halfbond=from_bond.halfbond, color=from_bond.color)

    from .bond import is_reasonable
    from chimerax.atomic import Residue
    bb_names = Residue.aa_min_ordered_backbone_names if chain.polymer_type == Residue.PT_AMINO \
        else Residue.na_min_ordered_backbone_names
    def get_bond_atoms(r1, r2, bb_names):
        if r1 and r2:
            a1 = r1.find_atom(bb_names[-1])
            if a1:
                a2 = r2.find_atom(bb_names[0])
                if a2:
                    return (a1, a2)
        return None
    if bond_start is not False:
        bond_atoms = get_bond_atoms(before, replacements[0], bb_names)
        if bond_atoms:
            if bond_start is True or is_reasonable(*bond_atoms):
                add_bond(*bond_atoms)
    if bond_end is not False:
        bond_atoms = get_bond_atoms(replacements[-1], after, bb_names)
        if bond_atoms:
            if bond_start is True or is_reasonable(*bond_atoms):
                add_bond(*bond_atoms)
    if reset_chain_info:
        from chimerax.atomic import Sequence
        residues = []
        characters = []
        for r in chain.residues:
            if r:
                residues.append(r)
                characters.append(Sequence.rname3to1(r.name))
        chain.bulk_set(residues, characters)


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
    from chimerax.atomic import AtomArg, ElementArg, StructureArg, AtomsArg, BondArg, ResiduesArg
    from chimerax.atomic.bond_geom import geometry_name
    desc = CmdDesc(
        required=[('atoms', AtomsArg)],
        synopsis = 'invert chirality of substituents'
    )
    register('build invert', desc, cmd_invert, logger=logger)

    desc = CmdDesc(
        required=[('atoms', AtomsArg)],
        keyword = [('length', PositiveFloatArg),
            ('omega', FloatArg), ('phi', FloatArg),
            ('move', EnumOf(("large", "small", "N", "C")))],
        synopsis = 'join models through peptide bond'
    )
    register('build join peptide', desc, cmd_join_peptide, logger=logger)

    desc = CmdDesc(
        required=[('atoms', AtomsArg)],
        keyword = [('length', PositiveFloatArg),
            ('dihedral', FloatArg), ('dihedral_atoms', AtomsArg),
            ('move', Or(StructureArg, EnumOf(("large", "small"))))],
        synopsis = 'join models through arbitrary bond'
    )
    register('build join bond', desc, cmd_join_bond, logger=logger)

    desc = CmdDesc(
        required=[('atom', AtomArg), ('element', ElementArg), ('num_bonds', IntArg)],
        keyword = [('geometry', EnumOf(range(len(geometry_name)), ids=geometry_name)),
            ('name', StringArg), ('connect_back', BoolArg), ('color_by_element', BoolArg),
            ('res_name', StringArg), ('new_res', BoolArg)],
        synopsis = 'modify atom'
    )
    register('build modify', desc, cmd_modify_atom, logger=logger)

    desc = CmdDesc(
        required=[('replaced_residues', ResiduesArg)],
        required_arguments = ['with_'],
        keyword = [('with_', ResiduesArg), ('numbering_start', IntArg),
            ('bond_start', BoolArg), ('bond_end', BoolArg)],
        synopsis = 'replace residues with information from other residues'
    )
    register('build replace', desc, cmd_replace_residues, logger=logger)

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
