# vim: set expandtab ts=4 sw=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
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
            residues[residues.chain_ids == chain_id].chain_ids = new_id

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
            raise UserError("Cannot find global psudobond group named '%s'" % name)
    else:
        pbg = a1.structure.pseudobond_group(name, create_type=None)
        if not pbg:
            raise UserError("Cannot find psudobond group named '%s' for structure %s" % (name, a1.structure))
    for pb in pbg.pseudobonds:
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
