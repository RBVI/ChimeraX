from chimera.core.structure import StructureModel


def osl_ident(i, atom_names, atom_residue_numbers, atom_chain_ids):
    # Chimera 1 style oslIdent without the model number
    return ':%s.%s@%s' % (atom_residue_numbers[i], atom_chain_ids[i],
                          atom_names[i])

for m in Chimera2_session.models.list():
    if not isinstance(m, StructureModel):
        continue
    an = m.mol_blob.atoms.names
    arn = m.mol_blob.atoms.residues.numbers
    arc = m.mol_blob.atoms.residues.chain_ids
    bond_indices = m.mol_blob.bond_indices

    atoms = [None] * len(an)
    for i in range(len(atoms)):
        atoms[i] = []
    for a0, a1 in bond_indices:
        atoms[a0].append(a1)
        atoms[a1].append(a0)

    for i in range(len(atoms)):
        bond_names = [osl_ident(j, an, arn, arc) for j in atoms[i]]
        bond_names.sort()
        if bond_names:
            print('%s: %s' % (osl_ident(i, an, arn, arc),
                  ' '.join(bond_names)))

raise SystemExit(0)
