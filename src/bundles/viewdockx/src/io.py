# vim: set expandtab shiftwidth=4 softtabstop=4:


def open_mol2(session, stream, name, auto_style):
    structures = []
    atoms = 0
    bonds = 0
    while True:
        s = _read_block(session, stream, auto_style)
        if not s:
            break
        structures.append(s)
        atoms += s.num_atoms
        bonds += s.num_bonds
    status = ("Opened mol2 file containing "
              "{} structures {} atoms, {} bonds".format
              (len(structures), atoms, bonds))
    return structures, status


def _read_block(session, stream, auto_style):
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

    from chimerax.core.atomic import AtomicStructure
    s = AtomicStructure(session, auto_style=auto_style)
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
    mol_labels = ["mol_name", ["num_atoms", "num_bonds", "num_subst",
                               "num_feat", "num_sets"],
                  "mol_type", "charge_type", "status_bits"]

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


def build_residues(s, subst_dict):
    """ create chimeraX substructure dictionary (csd) """
    csd = {}
    # csd will be something like {"1": <residue>}
    for s_index in subst_dict:
        # new_residue(residue_name, chain_id, pos)
        residue = s.new_residue(subst_dict[s_index][0][:4],
                                str(subst_dict[s_index][5]),
                                int(s_index))
        csd[s_index] = residue
    return csd


def build_atoms(s, csd, atom_dict):
    """ Creates chimeraX atom dictionary (cad)"""
    from numpy import array, float64
    cad = {}
    for key in atom_dict:
        name = atom_dict[key][0]
        element = atom_dict[key][4]
        if "." in element:
            split_element = element.split(".")
            element = split_element[0]
        xyz = [float(n) for n in atom_dict[key][1:4]]
        new_atom = s.new_atom(name, element)
        new_atom.coord = array(xyz, dtype=float64)
        # new_atom.serial_number = int(key)
        # adding new atom to subst_id 
        csd[atom_dict[key][5]].add_atom(new_atom)
        cad[key] = new_atom
    return cad


def build_bonds(s, cad, bond_dict):
    for key in bond_dict:
        atom1index = bond_dict[key][0]
        atom2index = bond_dict[key][1]
        try:
            a1 = cad[atom1index]
            a2 = cad[atom2index]
        except KeyError:
            print("Error : bad atom index in bond")
        else:
            s.new_bond(a1, a2)
