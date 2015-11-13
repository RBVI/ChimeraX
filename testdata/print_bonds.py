from chimera.core.structure import StructureModel


def osl_ident(chain_id, residue_number, atom_name):
    # Chimera style oslIdent without the model number
    return ':%s.%s@%s' % (residue_number, chain_id, atom_name)


for m in session.models.list():
    if not isinstance(m, StructureModel):
        continue
    bonds = {}
    a0s, a1s = m.mol_blob.bonds.atoms
    a0names = a0s.names
    a0rn = a0s.residues.numbers
    a0rc = a0s.residues.chain_ids
    a1names = a1s.names
    a1rn = a1s.residues.numbers
    a1rc = a1s.residues.chain_ids
    # print(len(a0names), 'bonds', flush=True)
    for i in range(len(a0names)):
        a0 = (a0rc[i], a0rn[i], a0names[i])
        a1 = (a1rc[i], a1rn[i], a1names[i])
        bonds.setdefault(a0, []).append(a1)
        bonds.setdefault(a1, []).append(a0)
    # print(len(bonds), 'atoms', flush=True)
    atoms = list(bonds)
    atoms.sort()
    for a in atoms:
        other_atoms = bonds[a]
        other_atoms.sort()
        print("%s: %s" % (osl_ident(*a),
                          ', '.join([osl_ident(*x) for x in other_atoms])))

session.logger.clear()
raise SystemExit(0)
