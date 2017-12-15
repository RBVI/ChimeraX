# vim: set expandtab shiftwidth=4 softtabstop=4:

# Mol2 specification: http://chemyang.ccnu.edu.cn/ccb/server/AIMMS/mol2.pdf


def open_mol2(session, stream, name, auto_style, atomic):
    structures = []
    atoms = 0
    bonds = 0
    while True:
        s = _read_block(session, stream, auto_style, atomic)
        if not s:
            break
        structures.append(s)
        atoms += s.num_atoms
        bonds += s.num_bonds
    status = ("Opened mol2 file containing "
              "{} structures {} atoms, {} bonds".format
              (len(structures), atoms, bonds))
    return structures, status


def _read_block(session, stream, auto_style, atomic):
    """Read all sections for a single entry and build Structure instance"""
    # First section should be comments
    # Second section: "@<TRIPOS>MOLECULE"
    # Third section: "@<TRIPOS>ATOM"
    # Fourth section: "@<TRIPOS>BOND"
    # Fifth section: "@<TRIPOS>SUBSTRUCTURE"
    data_dict, molecular_dict = read_com_and_mol(session, stream)
    if not molecular_dict:
        return None
    atom_dict = read_atom(session, stream)
    bond_dict = read_bond(session, stream)
    subst_dict = read_subst(session, stream) #pass in # of substructures
    if not subst_dict:
        subst_dict = {}
        subst_set = set()
        for i in atom_dict.values():
            subst_set.update([(i[5], i[6])])
        for i in subst_set:
            subst_dict[i[0]] = [i[1], None, None, None, None, '****']

    if atomic:
        from chimerax.core.atomic import AtomicStructure as StructureClass
    else:
        from chimerax.core.atomic import Structure as StructureClass
    s = StructureClass(session, auto_style=auto_style)
    s.name = molecular_dict["name"]
    _set_structure_attribute(s, molecular_dict, "charge_model")
    _set_structure_attribute(s, molecular_dict, "mol2_type")
    _set_structure_attribute(s, molecular_dict, "mol2_comment")
    csd = build_residues(s, subst_dict)
    cad = build_atoms(s, csd, atom_dict)
    build_bonds(s, cad, bond_dict)
    s.viewdockx_data = data_dict
    return s


def read_com_and_mol(session, stream):
    """Parses commented section"""
    # Comments section
    data_dict = {}
    while True:
        comment = stream.readline()
        if not comment: 
            break
        if not data_dict and comment[0] == "\n": #before the comment section
            continue
        if comment[0] != "#":  #for the end of comment section
            break
        line = comment.replace("#", "")
        parts = line.split(":")
        parts = [item.strip() for item in parts]
        if ":" not in line:
            for i in range(len(line), 1, -1):
                if line[i-1] == " ":
                    data_dict[line[:i].strip()] = line[i:].strip()
                    break
        else:
            data_dict[(parts[0])] = parts[1]

    # Molecule section
    if comment == "@<TRIPOS>MOLECULE":
        pass
    else:
        molecule_line = stream.readline()
        while "@<TRIPOS>MOLECULE" not in molecule_line:
            if not molecule_line: # Unexpected EOF
                return None, None
            molecule_line = stream.readline()

    molecular_dict = {}
    mol_labels = ["name", ["num_atoms", "num_bonds", "num_subst",
                           "num_feat", "num_sets"],
                  "mol2_type", "charge_model", "status_bits", "mol2_comment"]

    line_num = 0
    for label in mol_labels:
        line_num += 1
        last_pos = stream.tell()
        molecule_line = stream.readline().strip()
        if "@<TRIPOS>ATOM" in molecule_line:
            stream.seek(last_pos)
            break
        if line_num == 1:
            # Molecule name
            while not molecule_line:
                molecule_line = stream.readline().strip()
            molecular_dict[label] = molecule_line
            continue
        if line_num == 2:
            # Number of atoms, bonds, substructures, features and sets
            molecule_line = molecule_line.split()
            if all(isinstance(int(item), int) for item in molecule_line):
                molecular_dict.update(dict(zip(label, molecule_line)))
            else:
                raise ValueError("Second line needs to be series of integers")
            continue
        else:
            molecule_line = molecule_line.strip()
            molecular_dict[label] = molecule_line

    return data_dict, molecular_dict


def read_atom(session, stream):
    """parses atom section"""
    while "@<TRIPOS>ATOM" not in stream.readline():
        pass
    atom_dict = {}
    while True:
        last_pos = stream.tell()
        atom_line = stream.readline().strip()
        if not atom_line:
            stream.seek(last_pos)
            break
        if "@" in atom_line:
            stream.seek(last_pos)
            break
        if len(atom_line) == 0:
            print("error: no line found")
        parts = atom_line.split()
        if len(parts) not in range(6, 11):
            print("error: not enough or too many entries on a line")
            return None
        atom_dict[(parts[0])] = parts[1:]
    return atom_dict

def read_bond(session, stream):
    """parses bond section"""
    while "@<TRIPOS>BOND" not in stream.readline():
        pass
    bond_dict = {}
    while True:
        last_pos = stream.tell()
        bond_line = stream.readline()
        parts = bond_line.split()
        if not bond_line or "@" in bond_line or bond_line[0] == "#" or parts == 0:
            stream.seek(last_pos)
            break
        if len(parts) != 4:
            print("error: not enough entries in under bond data")
            raise ValueError
        if not isinstance(int(parts[0]), int):
            print("error: first value is needs to be an integer")
            raise ValueError
        bond_dict[parts[0]] = parts[1:3]
    return bond_dict

def read_subst(session, stream):
    """parses substructure section"""
    last_pos = stream.tell()
    subst_line = stream.readline()
    while "@<TRIPOS>SUBSTRUCTURE" not in subst_line:
        if "#" in subst_line or not subst_line:
            stream.seek(last_pos)
            return None

        subst_line = stream.readline()
    subst_dict = {}
    while True:
        last_pos = stream.tell()
        subst_line = stream.readline()
        parts = subst_line.split()
        if not subst_line or len(parts) == 0 or "#" in subst_line:
            stream.seek(last_pos)
            break
        if "#" in subst_line:
            stream.seek(last_pos)
            break
        subst_dict[parts[0]] = parts[1:]
    return subst_dict


def _set_structure_attribute(s, mol_dict, attr):
    v = mol_dict.get(attr, "").strip()
    if v:
        setattr(s, attr, v)


def build_residues(s, subst_dict):
    """ create chimeraX substructure dictionary (csd) """
    # subst_dict key is "subst_id"
    # Indices in subst_dict lists are:
    #   0: subst_name
    #   1: root_atom (matches "atom_id")
    #   2: subst_type [optional from here down]
    #   3: dict_type
    #   4: chain
    #   5: inter_bonds
    #   6: status
    #   7: comment
    csd = {}
    # csd will be something like {"1": <residue>}
    for subst_id, data_list in subst_dict.items():
        name = data_list[0][:4]
        try:
            chain = data_list[4]
            if chain == "****":
                raise IndexError("no chain")
        except IndexError:
            chain = ''
        # new_residue(residue_name, chain_id, pos)
        residue = s.new_residue(name, chain, int(subst_id))
        csd[subst_id] = residue
    return csd


def build_atoms(s, csd, atom_dict):
    """ Creates chimeraX atom dictionary (cad)"""
    # atom_dict key is "atom_id"
    # Indices in atom_dict lists are:
    #   0: atom_name
    #   1: x
    #   2: y
    #   3: z
    #   4: atom_type (e.g., C.3)
    #   5: subst_id [optional from here down]
    #   6: subst_name
    #   7: charge
    #   8: status_bit
    from numpy import array, float64
    cad = {}
    for atom_id, data_list in atom_dict.items():
        name = data_list[0]
        element = data_list[4]
        if "." in element:
            split_element = element.split(".")
            element = split_element[0]
        xyz = [float(n) for n in data_list[1:4]]
        new_atom = s.new_atom(name, element)
        new_atom.coord = array(xyz, dtype=float64)
        try:
            new_atom.charge = float(data_list[7])
        except (IndexError, ValueError):
            pass
        # new_atom.serial_number = int(key)
        # adding new atom to subst_id 
        csd[data_list[5]].add_atom(new_atom)
        cad[atom_id] = new_atom
    return cad


def build_bonds(s, cad, bond_dict):
    # bond_dict key is "bond_id"
    # Indices in bond_dict lists are:
    #   0: origin_atom_id
    #   1: target_atom_id
    #   2: bond_type
    #   3: status_bits [optional from here down]
    for bond_id, data_list in bond_dict.items():
        atom1index = data_list[0]
        atom2index = data_list[1]
        try:
            a1 = cad[atom1index]
            a2 = cad[atom2index]
        except KeyError:
            print("Error : bad atom index in bond")
        else:
            s.new_bond(a1, a2)
