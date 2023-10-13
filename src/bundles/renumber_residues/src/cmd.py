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

def cmd_renumber(session, residues, *, relative=True, start=None, seq_start=None):
    from chimerax.core.errors import UserError
    if residues is None:
        from chimerax.atomic import all_residues
        residues = all_residues(session)
    if not residues:
        raise UserError("No residues specified")

    if start is None and seq_start is None:
        seq_numbering = False
        start_number = 1
    elif start is None:
        seq_numbering = True
        start_number = seq_start
    elif seq_start is None:
        seq_numbering = False
        start_number = start
    else:
        raise UserError("Cannot specify both 'start' and seqStart' keywords")

    # verify no conflicts before making actual changes
    s_proposed_changes = {}
    s_changed_residues = {}
    change_info = []
    for s, s_residues in residues.by_structure:
        s_proposed_changes[s] = proposed_changes = set()
        s_changed_residues[s] = changed_residues = set()
        chain_ids = s_residues.unique_chain_ids
        for cid in chain_ids:
            chain_residues = sorted(s_residues[s_residues.chain_ids == cid])
            if seq_numbering:
                for r in chain_residues:
                    if r.chain:
                        seq_offset = r.chain.residues.index(r)
                        break
                else:
                    seq_offset = 0
            else:
                seq_offset = 0
            if relative:
                offset = (start_number + seq_offset) - chain_residues[0].number
            else:
                from itertools import count
                counter = count(start_number + seq_offset)
            for r in chain_residues:
                if relative:
                    new_number = r.number + offset
                    new_insertion_code = r.insertion_code
                else:
                    new_number = next(counter)
                    new_insertion_code = ''
                proposed_changes.add((cid, new_number, new_insertion_code))
                changed_residues.add(r)
                if r.number != new_number or r.insertion_code != new_insertion_code:
                    change_info.append((r, new_number, new_insertion_code))
        # check for conflicts
        for r in s.residues:
            if r in changed_residues:
                continue
            if (r.chain_id, r.number, r.insertion_code) in proposed_changes:
                from chimerax.core.errors import UserError
                raise UserError("Proposed renumbering conflicts with existing residue %s" % r)
    # apply changes
    for r, number, insertion_code in change_info:
        r.number = number
        r.insertion_code = insertion_code
    session.logger.info("%d residues renumbered" % len(change_info))

def register_command(command_name, logger):
    from chimerax.core.commands import CmdDesc, register, BoolArg, Or, EmptyArg, IntArg
    from chimerax.atomic import ResiduesArg
    desc = CmdDesc(
        required=[('residues', Or(ResiduesArg,EmptyArg))],
        keyword = [('relative', BoolArg), ('start', IntArg), ('seq_start', IntArg)],
        synopsis = 'Renumber residues'
    )
    register('renumber', desc, cmd_renumber, logger=logger)
