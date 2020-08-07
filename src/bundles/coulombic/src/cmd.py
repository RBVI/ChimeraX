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

chargeable_residues = set(['ILE', 'DG', 'DC', 'DA', 'GLY', 'ATP', 'TRP', 'DT', 'GLU', 'NH2', 'ASP', 'NAD', 'LYS', 'PRO', 'ASN', 'A', 'CYS', 'C', 'G', 'THR', 'HOH', 'GTP', 'HIS', 'U', 'NDP', 'SER', 'GDP', 'PHE', 'ALA', 'MET', 'ACE', 'NME', 'ADP', 'LEU', 'ARG', 'VAL', 'TYR', 'GLN', 'HID', 'HIP', 'HIE', 'MSE'])

def cmd_coulombic(session, atoms, *, surfaces=None, his_scheme=None):
    atoms_per_surf = []
    from chimerax.atomic import all_atomic_structures, MolecularSurface, all_atoms
    if atoms is None:
        if surfaces is None:
            # surface all chains
            for struct in all_atomic_structures(session):
                # don't create surface until charges checked
                if struct.num_chains == 0:
                    atoms_per_surf.append((struct.atoms, None))
                else:
                    for chain in struct.chains:
                        atoms_per_surf.append((chain.existing_residues.atoms, None))
        else:
            for srf in surfaces:
                if isinstance(srf, MolecularSurface):
                    atoms_per_surf.append((srf.atoms, srf))
                else:
                    atoms_per_surf.append((all_atoms(session), srf))
    else:
        if surfaces is None:
            # on a per-structure basis, determine if the atoms contain any polymers, and if so then
            # surface those chains (and not non-polymers); otherwise surface the atoms
            by_chain = {}
            for struct, chain_id, chain_atoms in atoms.by_chain:
                by_chain.setdefault(struct, {})[chain_id] = chain_atoms
            for struct, struct_atoms in atom.by_structure:
                try:
                    for chain_atoms in by_chain[struct].values():
                        atoms_per_surf.append((chain_atoms, None))
                except KeyError:
                    atoms_per_surf.append((struct_atoms, None))
        else:
            for srf in surfaces:
                atoms_per_surf.append((atoms, srf))

    # check whether the atoms have charges, and if not, that we know how to assign charges
    # to the requested atoms
    problem_residues = set()
    needs_assignment = set()
    for atoms, srf in atoms_per_surf:
        for r in atoms.unique_residues:
            if getattr(r, '_coulombic_his_scheme', his_scheme) != his_scheme:
                # should only be set on HIS residues
                needs_assignment.add(r)
            else:
                for a in r.atoms:
                    try:
                        a.charge + 1.0
                    except (AttributeError, TypeError):
                        if r.name in chargeable_residues:
                            needs_assignment.add(r)
                        else:
                            problem_residues.add(r.name)
                        break
    if problem_residues:
        from chimerax.core.commands import commas
        raise UserError("Don't know how to assign charges to the following residue types: %s"
            % commas(problem_residues, conjunction='and'))

    if needs_assignment:
        from .coulombic import assign_charges, ChargeError
        try:
            assign_charges(session, needs_assignment, his_scheme)
        except ChargeError as e:
            raise UserError(str(e))

def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, Or, EmptyArg, SurfacesArg, EnumOf
    from chimerax.atomic import AtomsArg
    desc = CmdDesc(
        required = [('atoms', Or(AtomsArg, EmptyArg))],
        keyword = [
            ('surfaces', SurfacesArg),
            ('his_scheme', EnumOf(['HIP', 'HIE', 'HID'])),
        ],
        synopsis = 'Color surfaces by coulombic potential'
    )
    register("coulombic", desc, cmd_coulombic, logger=logger)
