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
        atoms = 0
        bonds = 0  # Initialize bonds to 0 by default
        
        while line != '\n':
          tokens = line.split()
          
          if len(tokens) >= 2:
              if tokens[1] == 'atoms':
                atoms = int(tokens[0])
              elif tokens[1] == 'bonds':
                bonds = int(tokens[0])

          line = stream.readline()

        session.logger.info(f"LAMMPS data: {atoms} atoms {bonds} bonds")

        # SKIP UNTIL MASSES SECTION

        line = stream.readline()
        max_lines_to_search = 1000
        search_count = 0
        masses_section_found = False
        
        while search_count < max_lines_to_search:
            if not line:  # End of file
                break
            if line.startswith("Masses"):
                masses_section_found = True
                break
            line = stream.readline()
            search_count += 1
            
        if not masses_section_found:
            raise UserError("Masses section not found in DATA file")
            
        line = stream.readline() # SKIP BLANK LINE

        # PARSE MASSES

        masses = {}

        tokens = stream.readline().split()
        while tokens and tokens[0].isdigit():
          masses[int(tokens[0])] = float(tokens[1])
          tokens = stream.readline().split()

        # SKIP UNTIL ATOMS SECTION

        line = stream.readline()
        max_lines_to_search = 1000
        search_count = 0
        atoms_section_found = False
        
        while search_count < max_lines_to_search:
            if not line:  # End of file
                break
            if line.startswith("Atoms"):
                atoms_section_found = True
                break
            line = stream.readline()
            search_count += 1
            
        if not atoms_section_found:
            raise UserError("Atoms section not found in DATA file")
            
        line = stream.readline() # SKIP BLANK LINE

        # PARSE ATOMS

        atoms_list = []
        atoms_dict = {}
        
        # Determine atom style format based on first line of atoms section
        tokens = stream.readline().split()
        if not tokens:
            raise UserError("Empty Atoms section")
            
        # Determine the atom style based on the number of columns
        # Full atom style: id mol type q x y z [tx ty tz]
        # Charge atom style: id type q x y z
        num_columns = len(tokens)
        
        # Default positions for atom coordinates
        x_pos, y_pos, z_pos = 4, 5, 6
        mol_pos, type_pos = 1, 2
        
        # Adjust positions for atom_style charge (or other styles with fewer columns)
        if num_columns <= 7:  # Likely atom_style charge
            session.logger.info("Detected atom_style charge or similar format")
            mol_pos = 0  # Not present, will use atom id as molecule id
            type_pos = 1
            x_pos, y_pos, z_pos = 3, 4, 5
            
        while tokens:
          tag = int(tokens[0])
          
          # Handle molecule ID based on format
          if mol_pos == 0:
              # For atom_style charge, use atom id as molecule id
              mol = tag
          else:
              mol = int(tokens[mol_pos])
              
          type = int(tokens[type_pos])
          
          # Make sure we don't go out of bounds
          if x_pos >= len(tokens) or y_pos >= len(tokens) or z_pos >= len(tokens):
              raise UserError(f"Atom coordinates not found at expected positions. Atom line: {' '.join(tokens)}")
              
          xyz = array([float(tokens[x_pos]), float(tokens[y_pos]), float(tokens[z_pos])], dtype=float64)
          
          residue = structure.find_residue(" ", mol)
          if residue is None: residue = structure.new_residue(str(mol), " ", mol)
          
          # Check if the atom type exists in masses dictionary
          if type not in masses:
              raise UserError(f"Atom type {type} not found in Masses section")
              
          element = determine_element_from_mass(masses[type])
          atoms_list.append([tag, element, residue, xyz])
          tokens = stream.readline().split()

        atoms_list.sort(key=lambda atom:atom[0])

        for atom in atoms_list:
          atoms_dict[atom[0]] = add_atom(str(atom[0]), atom[1], atom[2], atom[3], serial_number=atom[0])

        # PROCESS BONDS SECTION IF BONDS EXIST
        if bonds > 0:
            # Try to find the Bonds section
            line = stream.readline()
            bonds_section_found = False
            
            # Limit the number of lines to search to avoid infinite loop
            max_lines_to_search = 100
            search_count = 0
            
            while search_count < max_lines_to_search:
                if not line:  # End of file
                    break
                if line.startswith("Bonds"):
                    bonds_section_found = True
                    break
                line = stream.readline()
                search_count += 1
            
            if bonds_section_found:
                line = stream.readline()  # SKIP BLANK LINE
                
                # PARSE BONDS
                tokens = stream.readline().split()
                while tokens:
                    tag1 = int(tokens[2])
                    tag2 = int(tokens[3])
                    # Check if both atoms exist before adding bond
                    if tag1 in atoms_dict and tag2 in atoms_dict:
                        add_bond(atoms_dict[tag1], atoms_dict[tag2])
                    else:
                        session.logger.warning(f"Skipping bond: atom {tag1} or {tag2} not found")
                    tokens = stream.readline().split()
            else:
                session.logger.info("No Bonds section found in the file, despite bond count > 0")

        stream.close()

    except Exception as e:
        print(traceback.format_exc())
        raise UserError("Problem reading/processing DATA file '%s': %s" % (os.path.realpath(stream.name), e))

    from .read_coords import read_coords
    read_coords(session, coords, structure, data_fmt.nicknames[0], replace=True, **kw)
    return [structure], ""

def read_dump(session, path, model):
    from numpy import array, float64
    from chimerax.core.errors import UserError

    try:
        stream = open_input(path, encoding='UTF-8')
        stream.readline()
        timestep = int(stream.readline().split()[0])
        stream.readline()
        num_atoms = int(stream.readline().split()[0])
        for j in range(4): stream.readline()

        # eg. ITEM: ATOMS id type mol x y z
        line = stream.readline()
        tokens = line.split()
        print("LAMMPS dump format: ", tokens[2:])
        
        # Check for required columns and set defaults in case they're missing
        required_columns = ['id', 'x', 'y', 'z']
        for col in required_columns:
            if col not in tokens:
                raise UserError(f"Required column '{col}' not found in DUMP file. Header: {line}")
        
        # Get column indices, set default values for optional columns
        index_id = tokens.index('id')-2
        
        # Handle optional columns with default values
        if 'type' in tokens:
            index_type = tokens.index('type')-2
        else:
            index_type = -1  # Not found, will use a default value
            
        if 'mol' in tokens:
            index_mol = tokens.index('mol')-2
        else:
            index_mol = -1  # Not found, will use atom id as molecule id
            
        index_x = tokens.index('x')-2
        index_y = tokens.index('y')-2
        index_z = tokens.index('z')-2

    coords_list = []
    done = False
    i = 0

    while not done:

      coords_list.append([])

      for j in range(num_atoms):
        tokens = stream.readline().split()
        
        # Handle required fields
        try:
            id = int(tokens[index_id])
        except (IndexError, ValueError):
            raise UserError(f"Could not parse atom ID from line: {' '.join(tokens)}")
            
        # Handle optional fields with defaults
        if index_type >= 0 and index_type < len(tokens):
            try:
                type = int(tokens[index_type])
            except ValueError:
                type = 1  # Default type
        else:
            type = 1  # Default type
            
        if index_mol >= 0 and index_mol < len(tokens):
            try:
                mol = int(tokens[index_mol])
            except ValueError:
                mol = id  # Use atom ID as molecule ID
        else:
            mol = id  # Use atom ID as molecule ID
            
        # Handle coordinates
        try:
            x = float(tokens[index_x])
            y = float(tokens[index_y])
            z = float(tokens[index_z])
        except (IndexError, ValueError):
            raise UserError(f"Could not parse atom coordinates from line: {' '.join(tokens)}")
            
        coords_list[i].append([id, x, y, z])

      # Sort by atom ID
      coords_list[i].sort(key=lambda atom: atom[0])
      i += 1
      
      # Check for next frame
      next_line = stream.readline()
      if next_line:
        try:
            # Skip to next frame's atoms
            for j in range(8): 
                stream.readline()
        except Exception:
            # End of file or format error
            done = True
      else:
        done = True

    coords = array(coords_list, dtype=float64)[:,:,1:]
    stream.close()
    return num_atoms, coords
except Exception as e:
    if 'stream' in locals() and stream is not None:
        stream.close()
    print(traceback.format_exc())
    raise UserError(f"Problem reading/processing DUMP file '{path}': {e}")
