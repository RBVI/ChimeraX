# vim: set expandtab ts=4 sw=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This particular file is part of the ChimeraX library. You can also
# redistribute and/or modify it under the terms of the GNU Lesser General
# Public License version 2.1 as published by the Free Software Foundation.
# For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
# EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
# LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
# VERSION 2.1
#
# This notice must be embedded in or attached to all copies, including partial
# copies, of the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

from chimerax.core.errors import UserError

def log_chains(session, structures=None):
    if structures is None:
        from chimerax.atomic import AtomicStructure
        structures = [s for s in session.models if isinstance(s, AtomicStructure)]
    for s in structures:
        s._report_chain_descriptions(session)

def combine_cmd(session, structures, *, close=False, model_id=None, name=None, retain_ids=False,
        add_to_session=True):

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

    if retain_ids:
        res_numbers = {}
        polymer_type = {}
        for s in structures:
            for r in s.residues:
                key = str(r.number) + r.insertion_code
                pot = res_numbers.setdefault(r.chain_id, set())
                if key in pot:
                    raise UserError("'retainIds' requires residues number / insertion code combos be unique"
                        " in each chain with the same ID (duplicate: %s)" % r.string(omit_structure=True))
                pot.add(key)
            for chain in s.chains:
                if polymer_type.setdefault(chain.chain_id, chain.polymer_type) != chain.polymer_type:
                    raise UserError("Cannot combine chains with different polymer types (chain %s)" %
                        chain.chain_id)
    combination = structures[0].copy(name)

    # Compute needed remapping of chain IDs
    seen_ids = set(combination.residues.unique_chain_ids)
    for s in structures[1:]:
        chain_id_mapping = {}
        if not retain_ids:
            chain_ids = sorted(s.residues.unique_chain_ids)
            for chain_id in chain_ids:
                # blank chain IDs don't "play nice" when combined; always remap them
                if chain_id in seen_ids or chain_id.isspace():
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

    if retain_ids:
        # combine same-ID chains
        by_id = {}
        for chain in combination.chains:
            by_id.setdefault(chain.chain_id, []).append(chain)
        def end_residues(chain):
            first = last = None
            for r in chain.residues:
                if r:
                    if first is None:
                        first = r
                    last = r
            return (first, last)
        for chains in by_id.values():
            if len(chains) == 1:
                continue
            sortable_data = []
            for chain in chains:
                first, last = end_residues(chain)
                sortable_data.append(((first.number, first.insertion_code), first, last))
            sortable_data.sort()
            left_end = sortable_data[0][-1]
            backbone_names = left_end.aa_min_ordered_backbone_names if left_end.chain.polymer_type \
                == left_end.PT_AMINO else left_end.na_min_ordered_backbone_names
            for _, right_start, right_end in sortable_data[1:]:
                missing = False
                for bb_name in reversed(backbone_names):
                    left_atom = left_end.find_atom(bb_name)
                    if left_atom:
                        break
                    missing = True
                else:
                    left_atom = left_end.atoms[0]
                for bb_name in backbone_names:
                    right_atom = right_start.find_atom(bb_name)
                    if right_atom:
                        break
                    missing = True
                else:
                    right_atom = right_start.atoms[0]
                if missing or right_start.number - left_end.number > 1:
                    pbg = combination.pseudobond_group(combination.PBG_MISSING_STRUCTURE)
                    pbg.new_pseudobond(left_atom, right_atom)
                else:
                    combination.new_bond(left_atom, right_atom)
                left_end = right_end

    # eliminate blanks IDs coming from structure 0
    chain_ids = set(combination.residues.unique_chain_ids)
    for chain_id in chain_ids:
        if chain_id.isspace():
            from chimerax.atomic import next_chain_id
            new_id = next_chain_id(chain_id)
            while new_id in chain_ids:
                new_id = next_chain_id(new_id)
            session.logger.info("Remapping chain ID '%s' in %s to '%s'" % (chain_id, structures[0], new_id))
            residues = combination.residues
            # cannot set polymeric chain ID directly from residue; set chains then remaining residues
            blank_residues = residues[residues.chain_ids == chain_id]
            blank_residues.chains.chain_ids = new_id
            blank_residues[blank_residues.chain_ids == chain_id].chain_ids = new_id

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
    if add_to_session:
        session.models.add([combination])
    return combination

def label_missing_cmd(session, structures, max_chains):
    from . import all_atomic_structures, Pseudobonds
    lm_group_name = 'missing-structure length labels'
    if structures is None:
        structures = all_atomic_structures(session)
    from chimerax.label.label3d import label, label_delete
    from chimerax.core.objects import Objects
    import math
    from chimerax.core.commands import plural_form
    for structure in structures:
        try:
            pbg = structure.pbg_map[structure.PBG_MISSING_STRUCTURE]
        except KeyError:
            continue
        show = structure.num_chains <= max_chains
        if show:
            for pb in pbg.pseudobonds:
                a1, a2 = pb.atoms
                r1, r2 = a1.residue, a2.residue
                # only label in-chain pseudobonds
                if r1.chain != r2.chain or r1.chain is None:
                    continue
                if r1.name == "UNK" or r2.name == "UNK":
                    # e.g. 3j5p
                    gap_size = abs(r1.number - r2.number) - 1
                else:
                    gap_size = abs(r1.chain.residues.index(r1) - r2.chain.residues.index(r2)) - 1
                if gap_size < 1:
                    continue
                label(session, Objects(pseudobonds=Pseudobonds([pb])),
                    text="%d %s" % (gap_size, plural_form(gap_size, "residue")),
                    height=math.log10(max(gap_size, 1))+1)
        else:
            label_delete(session, Objects(pseudobonds=pbg.pseudobonds))

from chimerax.core.colors import BuiltinColors
def pbond_cmd(session, atoms, *, color=BuiltinColors["slate gray"], current_coordset_only=False, dashes=6,
        global_=False, name="custom", radius=0.075, reveal=False, show_dist=False):

    if len(atoms) != 2:
        raise UserError("Must specify exactly 2 atoms to form pseudobond between; you specified %d"
            % len(atoms))
    a1, a2 = atoms

    if global_ or a1.structure != a2.structure:
        if current_coordset_only:
            raise UserError("Cannot create per-coordset pseudobonds for global pseudobond groups")
        pbg = session.pb_manager.get_group(name, create=True)
        if pbg.id is None:
            session.models.add([pbg])
    else:
        try:
            pbg = a1.structure.pseudobond_group(name,
                create_type=("per coordset" if current_coordset_only else "normal"))
        except TypeError:
            raise UserError("Pseudobond group '%s' already exists as a %sper-coordset group"
                % (name, ("non-" if current_coordset_only else "")))
    pbg.dashes = dashes
    pb = pbg.new_pseudobond(a1, a2)
    pb.color = color.uint8x4()
    pb.radius = radius
    if reveal:
        for end in atoms:
            if end.display:
                continue
            res_atoms = end.residue.atoms
            if end.is_side_chain:
                res_atoms.filter(res_atoms.is_side_chains == True).displays = True
            elif end.is_backbone():
                res_atoms.filter(res_atoms.is_backbones() == True).displays = True
            else:
                res_atoms.displays = True
    dist_monitor = session.pb_dist_monitor
    if show_dist:
        if pbg not in dist_monitor.monitored_groups:
            dist_monitor.add_group(pbg)
    else:
        if pbg in dist_monitor.monitored_groups:
            dist_monitor.remove_group(pbg)

def xpbond_cmd(session, atoms, *, global_=False, name="custom"):
    if len(atoms) != 2:
        raise UserError("Must specify exactly 2 atoms to delete pseudobond between; you specified %d"
            % len(atoms))
    a1, a2 = atoms

    if global_ or a1.structure != a2.structure:
        pbg = session.pb_manager.get_group(name, create=False)
        if not pbg:
            raise UserError("Cannot find global pseudobond group named '%s'" % name)
    else:
        pbg = a1.structure.pseudobond_group(name, create_type=None)
        if not pbg:
            raise UserError("Cannot find pseudobond group named '%s' for structure %s"
                % (name, a1.structure))
    for pb in pbg.pseudobonds[:]:
        if a1 in pb.atoms and a2 in pb.atoms:
            pbg.delete_pseudobond(pb)
            if pbg.num_pseudobonds == 0:
                session.models.close([pbg])
                break
    else:
        raise UserError("No pseudobond between %s and %s found for %s" % (a1, a2, pbg))

def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, create_alias, Or, EmptyArg, StringArg, BoolArg, \
        ModelIdArg, NoneArg, NoArg, ColorArg, NonNegativeIntArg, PositiveFloatArg
    from .args import AtomicStructuresArg, AtomsArg

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
            ('retain_ids', BoolArg),
        ],
        synopsis = 'Copy/combine structure models')
    register('combine', combine_desc, combine_cmd, logger=logger)

    pbond_desc = CmdDesc(
        required=[('atoms', AtomsArg)],
        keyword=[
            ('color', ColorArg),
            ('current_coordset_only', BoolArg),
            ('dashes', NonNegativeIntArg),
            ('global_', NoArg),
            ('name', StringArg),
            ('radius', PositiveFloatArg),
            ('reveal', BoolArg),
            ('show_dist', BoolArg),
        ],
        synopsis = 'Create pseudobond')
    register('pbond', pbond_desc, pbond_cmd, logger=logger)

    # Not allowed to reuse CmdDesc instances, so...
    xpbond_kw = {
        'required': [('atoms', AtomsArg)],
        'keyword': [
            ('global_', NoArg),
            ('name', StringArg),
        ],
        'synopsis': 'Delete pseudobond'
    }
    register('pbond delete', CmdDesc(**xpbond_kw), xpbond_cmd, logger=logger)
    register('~pbond', CmdDesc(**xpbond_kw), xpbond_cmd, logger=logger)

    label_missing_desc = CmdDesc(
        required=[
            ('structures', Or(AtomicStructuresArg,EmptyArg)),
            ('max_chains', NonNegativeIntArg),
        ],
        synopsis = 'Show/hide missing-structure pseudobond labels')
    register('label missing', label_missing_desc, label_missing_cmd, logger=logger)

