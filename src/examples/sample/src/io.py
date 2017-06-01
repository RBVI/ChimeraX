# vim: set expandtab shiftwidth=4 softtabstop=4:


def open_xyz(session, stream, name):
    structures = []
    atoms = 0
    bonds = 0
    while True:
        s = _read_block(session, stream)
        if not s:
            break
        structures.append(s)
        atoms += s.num_atoms
        bonds += s.num_bonds
    status = ("Opened XYZ file containing %d structures (%d atoms, %d bonds)" %
              (len(structures), atoms, bonds))
    return structures, status


def _read_block(session, stream):
    # First line should be an integer count of the number of
    # atoms in the block.  Each block gets turned into an
    # AtomicStructure instance.
    count_line = stream.readline()
    if not count_line:
        return None
    try:
        count = int(count_line)
    except ValueError:
        # XXX: Should emit an error message
        return None
    from chimerax.core.atomic import AtomicStructure
    s = AtomicStructure(session)

    # Next line is a comment line
    s.comment = stream.readline().strip()

    # There should be "count" lines of atoms.
    from numpy import array, float64
    residue = s.new_residue("UNK", 'A', 1)
    element_count = {}
    for n in range(count):
        atom_line = stream.readline()
        if not atom_line:
            # XXX: Should emit an error message
            return None
        parts = atom_line.split()
        if len(parts) != 4:
            # XXX: Should emit an error message
            return None
        # Extract available data
        element = parts[0]
        xyz = [float(v) for v in parts[1:]]

        # Convert to required initializers
        # XXX: May need to convert element to usable form
        n = element_count.get(element, 0) + 1
        name = element + str(n)
        element_count[element] = n

        # Create atom
        atom = s.new_atom(name, element)
        atom.coord = array(xyz, dtype=float64)
        residue.add_atom(atom)
    s.connect_structure([residue], [residue], [], [])
    s.new_atoms()   # tell structure it needs to update
    return s
