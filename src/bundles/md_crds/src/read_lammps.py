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

import traceback, time

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

        # FIND ATOMS SECTION
        line = stream.readline()
        max_lines_to_search = 1000
        search_count = 0
        atoms_section_found = False
        atom_style_comment = ""
        
        while search_count < max_lines_to_search:
            if not line:  # End of file
                break
            if line.startswith("Atoms"):
                atoms_section_found = True
                # Extract the atom style from the header comment if present
                if "#" in line:
                    atom_style_comment = line.split("#")[1].strip()
                    session.logger.info(f"Atom style from header: {atom_style_comment}")
                break
            line = stream.readline()
            search_count += 1
            
        if not atoms_section_found:
            raise UserError("Atoms section not found in DATA file")
            
        # Skip blank line after Atoms header
        line = stream.readline().strip()
        while not line and line != None:  # Skip any blank lines
            line = stream.readline().strip()
            if not line:
                break

        # Get the first atom line for format detection
        if not line:
            raise UserError("Empty Atoms section")
            
        first_atom_tokens = line.split()
        if not first_atom_tokens:
            raise UserError("Empty atom line")
            
        num_columns = len(first_atom_tokens)
        session.logger.info(f"Detected {num_columns} columns in atom data")

        # Format detection based on atom style comment in header or column analysis
        atom_id_pos = 0  # Always the first column
        atom_type_pos = 1  # Default to second column
        mol_id_pos = -1   # May not exist
        x_pos, y_pos, z_pos = -1, -1, -1
        
        # First check if we have an explicit atom style in header
        if atom_style_comment:
            if "charge/kk" in atom_style_comment:
                session.logger.info("Using atom_style charge/kk format from header")
                atom_type_pos = 1
                x_pos, y_pos, z_pos = 3, 4, 5
            elif "charge" in atom_style_comment:
                session.logger.info("Using atom_style charge format from header")
                atom_type_pos = 1
                x_pos, y_pos, z_pos = 3, 4, 5
            elif "full" in atom_style_comment:
                session.logger.info("Using atom_style full format from header")
                mol_id_pos = 1
                atom_type_pos = 2
                x_pos, y_pos, z_pos = 4, 5, 6
            elif "molecular" in atom_style_comment:
                session.logger.info("Using atom_style molecular format from header")
                mol_id_pos = 1
                atom_type_pos = 2
                x_pos, y_pos, z_pos = 3, 4, 5
            elif "atomic" in atom_style_comment:
                session.logger.info("Using atom_style atomic format from header")
                atom_type_pos = 1
                x_pos, y_pos, z_pos = 2, 3, 4
                
        # If no style comment or unrecognized, determine from data structure
        if x_pos == -1:  # Only if not already set from comment
            # Test if columns are numeric or floating point
            try:
                # For each typical position, try to convert to float or int
                # This helps identify which values are coords vs atom types
                
                # For atom_style full: id mol type q x y z
                # If 3rd column is int and columns 4-6 are float, likely full style
                if (num_columns >= 7 and 
                    is_int(first_atom_tokens[0]) and 
                    is_int(first_atom_tokens[1]) and 
                    is_int(first_atom_tokens[2]) and 
                    is_float(first_atom_tokens[4]) and 
                    is_float(first_atom_tokens[5]) and 
                    is_float(first_atom_tokens[6])):
                    session.logger.info("Detected atom_style full format")
                    mol_id_pos = 1
                    atom_type_pos = 2
                    x_pos, y_pos, z_pos = 4, 5, 6
                
                # For atom_style charge/kk: id type q x y z
                # If 2nd column is int and 3rd is float (charge) and columns 4-6 are float, likely charge/kk
                elif (num_columns >= 6 and 
                     is_int(first_atom_tokens[0]) and 
                     is_int(first_atom_tokens[1]) and 
                     is_float(first_atom_tokens[2]) and 
                     is_float(first_atom_tokens[3]) and 
                     is_float(first_atom_tokens[4]) and 
                     is_float(first_atom_tokens[5])):
                    session.logger.info("Detected atom_style charge or charge/kk format")
                    atom_type_pos = 1
                    x_pos, y_pos, z_pos = 3, 4, 5
                
                # For atom_style molecular: id mol type x y z
                # If first 3 columns are int and columns 3-5 are float, likely molecular
                elif (num_columns >= 6 and 
                     is_int(first_atom_tokens[0]) and 
                     is_int(first_atom_tokens[1]) and 
                     is_int(first_atom_tokens[2]) and 
                     is_float(first_atom_tokens[3]) and 
                     is_float(first_atom_tokens[4]) and 
                     is_float(first_atom_tokens[5])):
                    session.logger.info("Detected atom_style molecular format")
                    mol_id_pos = 1
                    atom_type_pos = 2
                    x_pos, y_pos, z_pos = 3, 4, 5
                
                # For atom_style atomic: id type x y z
                # If first 2 columns are int and columns 2-4 are float, likely atomic
                elif (num_columns >= 5 and 
                     is_int(first_atom_tokens[0]) and 
                     is_int(first_atom_tokens[1]) and 
                     is_float(first_atom_tokens[2]) and 
                     is_float(first_atom_tokens[3]) and 
                     is_float(first_atom_tokens[4])):
                    session.logger.info("Detected atom_style atomic format")
                    atom_type_pos = 1
                    x_pos, y_pos, z_pos = 2, 3, 4
                
                # If we couldn't determine format, fall back to best guess
                else:
                    session.logger.warning("Could not definitively determine atom style, making best guess based on column count")
                    if num_columns >= 7:
                        # Assume full style with 7+ columns
                        mol_id_pos = 1
                        atom_type_pos = 2
                        x_pos, y_pos, z_pos = 4, 5, 6
                    elif num_columns >= 6:
                        # Assume charge style with 6+ columns
                        atom_type_pos = 1
                        x_pos, y_pos, z_pos = 3, 4, 5
                    else:
                        # Assume atomic style with minimal columns
                        atom_type_pos = 1
                        x_pos, y_pos, z_pos = 2, 3, 4
                        
            except Exception as e:
                # If we encounter any error in format detection, use a simple heuristic
                session.logger.warning(f"Error during format detection: {e}, using fallback format")
                if num_columns >= 7:
                    mol_id_pos = 1
                    atom_type_pos = 2
                    x_pos, y_pos, z_pos = 4, 5, 6
                elif num_columns >= 6:
                    atom_type_pos = 1
                    x_pos, y_pos, z_pos = 3, 4, 5
                else:
                    atom_type_pos = 1
                    x_pos, y_pos, z_pos = 2, 3, 4

        # Now process atoms with the determined format
        atoms_list = []
        atoms_dict = {}
        tokens = first_atom_tokens  # Start with the first line we already read

        while tokens:
            # Always get atom ID from first column
            atom_id = safe_int(tokens[atom_id_pos], fallback=len(atoms_list)+1)
            
            # Get molecule ID if available, otherwise use atom ID
            if mol_id_pos >= 0 and mol_id_pos < len(tokens):
                mol_id = safe_int(tokens[mol_id_pos], fallback=atom_id)
            else:
                mol_id = atom_id  # Default to atom ID if no molecule ID
            
            # Get atom type - safely parse
            if atom_type_pos < len(tokens):
                atom_type = safe_int(tokens[atom_type_pos], fallback=1)
            else:
                atom_type = 1  # Default type
            
            # Get coordinates - with safe parsing
            x = safe_float(tokens[x_pos] if x_pos < len(tokens) else "0", fallback=0.0)
            y = safe_float(tokens[y_pos] if y_pos < len(tokens) else "0", fallback=0.0)
            z = safe_float(tokens[z_pos] if z_pos < len(tokens) else "0", fallback=0.0)
            
            # Create coordinates array
            xyz = array([x, y, z], dtype=float64)
            
            # Get or create residue
            residue = structure.find_residue(" ", mol_id)
            if residue is None:
                residue = structure.new_residue(str(mol_id), " ", mol_id)
            
            # Check if atom type exists in masses dictionary
            if atom_type not in masses:
                session.logger.warning(f"Atom type {atom_type} not found in Masses section, using default element")
                element = "X"  # Use unknown element as fallback
            else:
                element = determine_element_from_mass(masses[atom_type])
            
            # Add atom to the list
            atoms_list.append([atom_id, element, residue, xyz])
            
            # Read next line
            line = stream.readline()
            tokens = line.split() if line else []

        # Sort atoms by ID and add to structure
        atoms_list.sort(key=lambda atom: atom[0])
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
                    if len(tokens) >= 4:  # Ensure we have enough columns
                        # Most bond formats have: bond_id bond_type atom1 atom2
                        tag1 = safe_int(tokens[2], fallback=0)
                        tag2 = safe_int(tokens[3], fallback=0)
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

# Helper functions for safe parsing
def is_int(val):
    try:
        int(val)
        return True
    except ValueError:
        return False

def is_float(val):
    try:
        float(val)
        return True
    except ValueError:
        return False

def safe_int(val, fallback=0):
    try:
        return int(val)
    except ValueError:
        return fallback

def safe_float(val, fallback=0.0):
    try:
        return float(val)
    except ValueError:
        return fallback

def read_dump2(session, path, model):
    from numpy import array, float64
    from chimerax.core.errors import UserError
    
    start = time.perf_counter()
    
    session.logger.info("*** read_dump()")

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
        
        elapsed = time.perf_counter() - start
        session.logger.info(f"*** read_dump() {elapsed:.6f} seconds")
        
        return num_atoms, coords
        
    except Exception as e:
        if 'stream' in locals() and stream is not None:
            stream.close()
        print(traceback.format_exc())
        raise UserError(f"Problem reading/processing DUMP file '{path}': {e}")












































import numpy as np
import multiprocessing as mp
import os
import time
import traceback

def parse_frame_worker(path, offset, num_atoms, num_cols, col_indices):
    """
    Worker jumps to byte offset and parses atom block.
    """
    with open(path, 'rb') as f:
        f.seek(offset)
        # Skip the 9 header lines to reach coordinate data
        for _ in range(9):
            f.readline()
        
        # Binary block read: significantly faster than text mode
        data = np.fromfile(f, dtype=np.float32, sep=' ', count=num_atoms * num_cols)
        data = data.reshape((num_atoms, num_cols))
        
        # Sort by atom ID column
        data = data[data[:, col_indices[0]].argsort()]
        
        # Return XYZ columns as float64 for ChimeraX
        return data[:, col_indices[1:]].astype(np.float64)

def read_dump(session, path, model, num_cores=None):
    from chimerax.core.errors import UserError
    import time
    
    start_time = time.perf_counter()
    if num_cores is None:
        num_cores = mp.cpu_count()

    try:
        offsets = []
        file_size = os.path.getsize(path)
        
        with open(path, 'rb') as f:
            # 1. INITIAL METRICS
            f.readline() # ITEM: TIMESTEP
            f.readline() # timestep value
            f.readline() # ITEM: NUMBER OF ATOMS
            num_atoms = int(f.readline().strip())
            for _ in range(4): f.readline() 
            
            # Correctly identify the ITEM: ATOMS header line
            header_line = f.readline().decode('utf-8')
            header_end_pos = f.tell()
            
            # Sample 3 lines for a robust average line length heuristic
            sample_lens = [len(f.readline()) for _ in range(3)]
            avg_line_len = sum(sample_lens) / 3.0
            
            # 2. COLUMN MAPPING
            tokens = header_line.split()
            col_names = tokens[2:] 
            num_cols = len(col_names)
            col_map = {name: i for i, name in enumerate(col_names)}
            col_indices = [col_map['id'], col_map['x'], col_map['y'], col_map['z']]

            # 3. CONSERVATIVE JUMP SEARCH
            est_frame_size = header_end_pos + (num_atoms * avg_line_len)
            offsets.append(0)
            print(f"*** Frame 0 offset: 0")
            
            # Start very conservatively (50% of first frame) to avoid skipping Frame 2
            search_pos = int(est_frame_size * 0.5) 

            while search_pos < file_size:
                f.seek(max(0, search_pos))
                # 256KB buffer provides a much larger landing zone to catch missed tags
                buffer = f.read(262144) 
                if not buffer:
                    break
                
                tag_idx = buffer.find(b"ITEM: TIMESTEP")
                
                if tag_idx != -1:
                    actual_offset = f.tell() - len(buffer) + tag_idx
                    
                    if actual_offset > offsets[-1]:
                        offsets.append(actual_offset)
                        if len(offsets) % 100 == 0 or len(offsets) < 10: # Reduce print noise
                            print(f"*** Frame {len(offsets)-1} offset: {actual_offset}")
                        
                        # Calculate actual size of last frame
                        last_frame_size = offsets[-1] - offsets[-2]
                        # Jump to 90% of the last frame size (Safer than 95% or 98%)
                        search_pos = actual_offset + int(last_frame_size * 0.90)
                    else:
                        # If we found the same tag, we must move the pointer forward
                        search_pos = actual_offset + 100 
                else:
                    # If tag not found, the jump was too far or the buffer too small
                    # Scan forward in small steps
                    search_pos += 65536

        session.logger.info(f"Jump-indexed {len(offsets)} frames. Parsing with {num_cores} cores.")

        # 4. PARALLEL PARSING
        with mp.Pool(processes=num_cores) as pool:
            args = [(path, off, num_atoms, num_cols, col_indices) for off in offsets]
            results = pool.starmap(parse_frame_worker, args)

        elapsed = time.perf_counter() - start_time
        session.logger.info(f"*** read_dump() {elapsed:.6f} seconds")

        return num_atoms, np.stack(results)

    except Exception as e:
        print(traceback.format_exc())
        raise UserError(f"Fast read_dump failed: {e}")
        
        
        
