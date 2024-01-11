# vim: set expandtab shiftwidth=4 softtabstop=4:


def open_xyz(session, stream):
    """Read an XYZ file from a file-like object.

    Returns the 2-tuple return value expected by the
    "open command" manager's :py:meth:`run_provider` method.
    """
    structures = []
    line_number = 0
    atoms = 0
    bonds = 0
    while True:
        s, line_number = _read_block(session, stream, line_number)
        if not s:
            break
        structures.append(s)
        atoms += s.num_atoms
        bonds += s.num_bonds
    status = ("Opened XYZ file containing %d structures (%d atoms, %d bonds)" %
              (len(structures), atoms, bonds))
    return structures, status


def _read_block(session, stream, line_number):
    # XYZ files are stored in blocks, with each block representing
    # a set of atoms.  This function reads a single block
    # and builds a ChimeraX AtomStructure instance containing
    # the atoms listed in the block.

    # First line should be an integer count of the number of
    # atoms in the block.
    count_line = stream.readline()
    if not count_line:
        # Reached EOF, normal termination condition
        return None, line_number
    line_number += 1
    try:
        count = int(count_line)
    except ValueError:
        session.logger.error("line %d: atom count missing" % line_number)
        return None, line_number

    # Create the AtomicStructure instance for atoms in this block.
    # All atoms in the structure are placed in one residue
    # since XYZ format does not partition atoms into groups.
    from chimerax.atomic import AtomicStructure
    from numpy import array, float64
    s = AtomicStructure(session)
    residue = s.new_residue("UNK", 'A', 1)

    # XYZ format supplies the atom element type only, but
    # ChimeraX keeps track of both the element type and
    # a unique name for each atom.  To construct the unique
    # atom name, the # 'element_count' dictionary is used
    # to track the number of atoms of each element type so far,
    # and the current count is used to build unique atom names.
    element_count = {}

    # Next line is a comment line
    s.comment = stream.readline().strip()
    line_number += 1

    # import convenience function for adding atoms
    from chimerax.atomic.struct_edit import add_atom

    # There should be "count" lines of atoms.
    for n in range(count):
        atom_line = stream.readline()
        if not atom_line:
            session.logger.error("line %d: atom data missing" % line_number)
            return None, line_number
        line_number += 1

        # Extract available data
        parts = atom_line.split()
        if len(parts) != 4:
            session.logger.error("line %d: atom data malformatted"
                                 % line_number)
            return None, line_number

        # Convert to required parameters for creating atom.
        # Since XYZ format only required atom element, we
        # create a unique atom name by putting a number after
        # the element name.
        xyz = [float(v) for v in parts[1:]]
        element = parts[0]
        n = element_count.get(element, 0) + 1
        name = element + str(n)
        element_count[element] = n

        # Create atom in AtomicStructure instance 's',
        # set its coordinates, and add to residue
        atom = add_atom(name, element, residue, array(xyz, dtype=float64))

    # Use AtomicStructure method to add bonds based on interatomic distances
    s.connect_structure()

    # Return AtomicStructure instance and current line number
    return s, line_number
