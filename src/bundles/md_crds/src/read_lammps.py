# vim: set expandtab shiftwidth=4 softtabstop=4:

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

import traceback

from chimerax.atomic import AtomicStructure
from chimerax.io import open_input
from .util import determine_element_from_mass, prep_coords

def read_data(session, stream, file_name, *, auto_style=True, coords=None, **kw):
    from chimerax.core.errors import UserError
    import os
    coords, data_fmt = prep_coords(session, coords, stream, "DATA", file_type="DUMP")

    structure = AtomicStructure(session, name=os.path.basename(file_name), auto_style=auto_style)

    try:
        from chimerax.atomic.struct_edit import add_atom, add_bond
        from numpy import array, float64

        stream.readline()
        stream.readline()

        # READ NUMBER OF ATOMS AND BONDS

        line = stream.readline()
        while line != '\n':
          tokens = line.split()

          if tokens[1] == 'atoms':
            atoms = int(tokens[0])
          elif tokens[1] == 'bonds':
            bonds = int(tokens[0])

          line = stream.readline()

        session.logger.info( f"LAMMPS data: {atoms} atoms {bonds} bonds")

        # SKIP UNTIL MASSES SECTION

        line = stream.readline()
        while not line.startswith("Masses"): line = stream.readline()
        line = stream.readline() # SKIP BLANK LINE

        # PARSE MASSES

        masses = {}

        tokens = stream.readline().split()
        while tokens and tokens[0].isdigit():
          masses[int(tokens[0])] = float(tokens[1])
          tokens = stream.readline().split()

        # SKIP UNTIL ATOMS SECTION

        line = stream.readline()
        while not line.startswith("Atoms"): line = stream.readline()
        line = stream.readline() # SKIP BLANK LINE

        # PARSE ATOMS

        atoms_list = []
        atoms = {}

        tokens = stream.readline().split()
        while tokens:
          tag = int(tokens[0])
          mol = int(tokens[1])
          type = int(tokens[2])
          xyz = array([float(tokens[4]),float(tokens[5]),float(tokens[6])], dtype=float64)
          residue = structure.find_residue(" ", mol)
          if residue is None: residue = structure.new_residue(str(mol), " ", mol)
          element = determine_element_from_mass(masses[type])
          atoms_list.append([tag, element, residue, xyz])
          tokens = stream.readline().split()

        atoms_list.sort(key=lambda atom:atom[0])

        for atom in atoms_list:
          atoms[atom[0]] = add_atom(str(atom[0]), atom[1], atom[2], atom[3], serial_number=atom[0])

        # SKIP UNTIL BONDS SECTION

        line = stream.readline()
        while not line.startswith("Bonds"): line = stream.readline()
        line = stream.readline() # SKIP BLANK LINE

        # PARSE BONDS

        tokens = stream.readline().split()
        while tokens:
          tag1 = int(tokens[2])
          tag2 = int(tokens[3])
          # FIXME: handle tag1 and/or tag2 not found
          add_bond(atoms[tag1], atoms[tag2])
          tokens = stream.readline().split()

        stream.close()

    except Exception as e:
        print(traceback.format_exc())
        raise UserError("Problem reading/processing DATA file '%s': %s" % (os.path.realpath(stream.name), e))

    from .read_coords import read_coords
    read_coords(session, coords, structure, data_fmt.nicknames[0], replace=True, **kw)
    return [structure], ""

def read_dump(session, path, model):
    from numpy import array, float64

    stream = open_input(path, encoding='UTF-8')
    stream.readline()
    timestep = int(stream.readline().split()[0])
    stream.readline()
    num_atoms = int(stream.readline().split()[0])
    for j in range(4): stream.readline()

    # eg. ITEM: ATOMS id type mol x y z
    tokens = stream.readline().split()
    print("LAMMPS dump format: ", tokens[2:])
    index_id = tokens.index('id')-2
    index_type = tokens.index('type')-2
    index_mol = tokens.index('mol')-2
    index_x = tokens.index('x')-2
    index_y = tokens.index('y')-2
    index_z = tokens.index('z')-2

    coords_list = []
    done = False
    i = 0

    while not done:

      coords_list.append([])

      for j in range(num_atoms):
        # FIXME: handle dump format other than id type mol x y z
        tokens = stream.readline().split()
        id = int(tokens[index_id])
        type = int(tokens[index_type])
        mol = int(tokens[index_mol])
        x,y,z = float(tokens[index_x]),float(tokens[index_y]),float(tokens[index_z])
        coords_list[i].append([id,x,y,z])

      coords_list[i].sort(key=lambda atom:atom[0])
      i += 1
      if stream.readline():
        for j in range(8): stream.readline()
      else:
        done = True

    coords = array(coords_list, dtype=float64)[:,:,1:]
    stream.close()
    return num_atoms, coords
