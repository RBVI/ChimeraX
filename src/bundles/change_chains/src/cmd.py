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

def cmd_change_chains(session, residues, from_ids, to_ids=None):
    from chimerax.core.errors import UserError
    # if to_ids is None, then map all residues to the (single element of) from_ids
    if to_ids is None:
        if len(from_ids) != 1:
            raise UserError("Chain ID argument must either two lists of IDs, or one single ID")
    elif len(from_ids) != len(to_ids):
        raise UserError('Number of "from" chain IDs (%d) does not match number of "to" chain IDs (%d)'
            % (len(from_ids), len(to_ids)))
    if residues is None:
        from chimerax.atomic import all_residues
        residues = all_residues(session)
    if not residues:
        raise UserError("No residues specified")
    if to_ids is not None:
        from numpy import array, isin
        unique_from_ids = array(from_ids)
        residues = residues[isin(residues.chain_ids, unique_from_ids)]
        if not residues:
            raise UserError("No residues specified have any of the 'from' chain IDs")

    # Verify that any polymeric chains are being changed in their entirety
    for chain in residues.unique_chains:
        full_chain_residues = chain.existing_residues
        if len(full_chain_residues) > len(full_chain_residues & residues):
            raise UserError("Cannot reassign chain ID to only part of polymeric chain (%s)" % chain)

    if to_ids is not None:
        chain_mapping = {}
        for from_id, to_id in zip(from_ids, to_ids):
            if from_id in chain_mapping:
                if chain_mapping[from_id] != to_id:
                    raise UserError("Cannot request mapping from one chain ID to two different IDs (%s"
                        "\N{RIGHTWARDS ARROW}%s,%s" % (from_id, chain_mapping[from_id], to_id))
            else:
                chain_mapping[from_id] = to_id

    # verify no conflicts before making actual changes
    change_info = []
    if to_ids is None:
        remapped_chain_id = from_ids[0]
    for s, s_residues in residues.by_structure:
        proposed_changes = set()
        changed_residues = set()
        for r in s_residues:
            if to_ids is None:
                if r.chain_id == remapped_chain_id:
                    continue
            else:
                try:
                    remapped_chain_id = chain_mapping[r.chain_id]
                except KeyError:
                    continue
            change = (remapped_chain_id, r.number, r.insertion_code)
            if change in proposed_changes:
                raise UserError("Proposed chain ID change would produce multiple residues with the same"
                    "chain-ID/number/insertion-code combo (%s/%d/%s)" % change)
            proposed_changes.add((remapped_chain_id, r.number, r.insertion_code))
            changed_residues.add(r)
            change_info.append((r, remapped_chain_id))
        # check for conflicts
        for r in s.residues:
            if r in changed_residues:
                continue
            if (r.chain_id, r.number, r.insertion_code) in proposed_changes:
                from chimerax.core.errors import UserError
                raise UserError("Proposed chainID change conflicts with existing residue %s" % r)
    # apply changes
    for r, cid in change_info:
        if r.chain_id != cid:
            if r.chain:
                r.chain.chain_id = cid
            else:
                r.chain_id = cid
    session.logger.info("Chain IDs of %d residues changed" % len(change_info))

def cmd_change_glys(session, chains=None):
    from chimerax.core.errors import UserError
    if chains is None:
        from chimerax.atomic import all_atomic_structures
        chains = all_atomic_structures(session).chains
    if not chains:
        raise UserError("No chains specified as base for changing")
    need_check = checked = set(chains.existing_residues)
    num_changed = 0
    while need_check:
        next_check = set()
        for r in need_check:
            for nb in r.neighbors:
                if nb in checked or nb in need_check or nb in next_check or nb.polymer_type != r.PT_NONE:
                    continue
                next_check.add(nb)
                if nb.chain_id != r.chain_id:
                    nb.chain_id = r.chain_id
                    num_changed += 1
            checked.add(r)
        need_check = next_check
    session.logger.info("Chain IDs of %d residues changed" % num_changed)

def register_command(command_name, logger):
    from chimerax.core.commands import CmdDesc, register, Or, EmptyArg, StringArg, ListOf
    from chimerax.atomic import ResiduesArg, UniqueChainsArg
    desc = CmdDesc(
        required=[('residues', Or(ResiduesArg,EmptyArg)), ('from_ids', ListOf(StringArg))],
        optional=[('to_ids', ListOf(StringArg))],
        synopsis = 'Change chain IDs'
    )
    register('changechains', desc, cmd_change_chains, logger=logger)

    desc = CmdDesc(
        optional=[('chains', UniqueChainsArg)],
        synopsis = 'Change chain IDs of glycosylations to match base chain ID'
    )
    register('changechains glycosylations', desc, cmd_change_glys, logger=logger)
