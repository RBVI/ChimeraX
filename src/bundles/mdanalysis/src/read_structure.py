# vim: set expandtab shiftwidth=4 softtabstop=4:

import os
import numpy as np
import MDAnalysis as mda
from chimerax.atomic import Element
from chimerax.core.errors import UserError, CancelOperation

def read_structure(session, path, file_name, format_name=None, *, auto_style=True, coords=None, **kw):
    """
    Unified structure reader using MDAnalysis.
    Handles PSF, GRO, LAMMPS Data, and other MDA-supported formats.
    """
    # 1. Prepare file arguments
    if coords:
        coords_path, _ = prep_coords(session, coords, path, format_name)
        load_args = [path, coords_path]
    else:
        coords_path = None
        load_args = [path]

    # 2. Dynamically build format arguments
    kwargs = {}
    fmt_map = {
        "psf": "PSF",
        "gro": "GRO",
        "data": "DATA",
        "pdb": "PDB"
    }
    mda_fmt = fmt_map.get(format_name, None)
    
    if mda_fmt:
        # Always tell MDA the topology format
        kwargs['topology_format'] = mda_fmt
        
        # Only force the coordinate format if we know this file natively contains coords.
        # This prevents MDA from crashing if coords=None on a pure PSF file.
        if coords is None and mda_fmt in ["GRO", "DATA", "PDB"]:
            kwargs['format'] = mda_fmt

    # If loading a LAMMPS dump as coordinates, inject the flags needed to read unwrapped image columns
    if coords_path:
        c_str = str(coords_path).lower()
        if c_str.endswith('.dump') or c_str.endswith('.lammpstrj'):
            kwargs['format'] = 'LAMMPSDUMP'
            kwargs['unwrap_images'] = False
            kwargs['additional_columns'] = ['ix', 'iy', 'iz']

    kwargs.update(kw)

    try:
        session.logger.status(f"Loading topology via MDAnalysis...")
        universe = mda.Universe(*load_args, **kwargs)
    except Exception as e:
        session.logger.warning(f"MDAnalysis failed to load {path}: {e}")
        raise UserError(f"Could not read file {file_name} via MDAnalysis: {e}")

    # 3. Build ChimeraX topology
    try:
        name = os.path.basename(file_name)
        model = universe_to_atomic_structure(session, universe, name, auto_style=auto_style)
    except Exception as e:
         raise UserError(f"Failed to convert MDAnalysis topology to ChimeraX structure: {e}")

    # 4. Process Coordinates and Trajectory
    msg = f"Imported {model.num_atoms} atoms, {len(universe.trajectory)} frames."
    if len(universe.trajectory) > 0:
        for i, timestep in enumerate(universe.trajectory):
            xyz = timestep.positions.astype(np.float64)

            # Apply LAMMPS image flags to generate unwrapped absolute coordinates if they exist
            if 'ix' in timestep.data and 'iy' in timestep.data and 'iz' in timestep.data:
                if timestep.dimensions is not None:
                    box = timestep.dimensions[:3]
                    xyz[:, 0] += timestep.data['ix'] * box[0]
                    xyz[:, 1] += timestep.data['iy'] * box[1]
                    xyz[:, 2] += timestep.data['iz'] * box[2]

            session.logger.status(f"Processing timestep {timestep.frame}")
            
            # --- THE FIX ---
            if i == 0:
                # Update the actual atoms' coordinates directly
                model.atoms.coords = xyz 
            else:
                # Append subsequent trajectory frames (i+1 to match ChimeraX 1-based indexing)
                model.add_coordset(id=i+1, xyz=xyz)

    return [model], msg


def determine_element_from_mass(mass, *, consider_hydrogens=True):
    """Guess element from atomic mass."""
    H = Element.get_element('H')
    nearest = None
    for high in range(1, Element.NUM_SUPPORTED_ELEMENTS+1):
        if Element.get_element(high).mass > mass:
            break
    else:
        high = Element.NUM_SUPPORTED_ELEMENTS

    if high == 1:
        return H

    max_hyds = 6 if consider_hydrogens else 0

    for num_hyds in range(max_hyds+1):
        adj_mass = mass - num_hyds * H.mass
        low_mass = Element.get_element(high-1).mass
        while low_mass > adj_mass and high > 1:
            high -= 1
            low_mass = Element.get_element(high-1).mass
        high_mass = Element.get_element(high).mass
        
        low_diff = abs(adj_mass - low_mass)
        high_diff = abs(adj_mass - high_mass)
        
        if low_diff < high_diff:
            diff = low_diff
            element = high-1
        else:
            diff = high_diff
            element = high
            
        if nearest is None or diff < nearest[1]:
            nearest = (element, diff)
            
    return Element.get_element(nearest[0])


def prep_coords(session, coords_file, input_path, format_name, *, file_type="coordinates"):
    """Helper to handle file dialogs if coords_file is missing."""
    if coords_file is None:
        if session.ui.is_gui and not session.in_script:
            if isinstance(input_path, str):
                path = input_path
            elif hasattr(input_path, 'name'):
                path = os.path.dirname(os.path.realpath(input_path.name))
            else:
                path = os.getcwd()
                
            from Qt.QtWidgets import QFileDialog
            from chimerax.core.errors import CancelOperation
            coords, _ = QFileDialog.getOpenFileName(
                caption=f"Specify {file_type} file for {format_name}",
                directory=path, options=QFileDialog.DontUseNativeDialog)
            if not coords:
                raise CancelOperation(f"No coordinates file specified for {format_name}")
            session.logger.info("Coordinates file: %s" % coords)
        else:
            from chimerax.core.errors import UserError
            raise UserError("'coords' keyword with coordinate-file argument must be supplied")
    else:
        coords = coords_file
    
    try:
        from chimerax.data_formats import NoFormatError # <-- THE MISSING IMPORT
        data_fmt = session.data_formats.open_format_from_file_name(coords)
    except NoFormatError:
        data_fmt = None 
        
    return coords, data_fmt


def universe_to_atomic_structure(session, u, name, auto_style=True):
    """
    Converts an MDAnalysis Universe to a ChimeraX AtomicStructure.
    Uses flat iteration over u.atoms to guarantee coordinate array alignment.
    """
    # Deferred imports to prevent ChimeraX from silently failing the bundle load
    import tinyarray
    from chimerax.atomic import AtomicStructure
    from chimerax.atomic.struct_edit import add_atom, add_bond

    s = AtomicStructure(session, name=name, auto_style=auto_style)
    
    # Pre-calculate elements if missing in topology
    elements = []
    has_elements = hasattr(u.atoms, 'elements') and not all(e == '' for e in u.atoms.elements)
    
    for atom in u.atoms:
        if has_elements and atom.element:
            try:
                el = Element.get_element(atom.element)
            except KeyError:
                el = determine_element_from_mass(atom.mass)
        else:
            el = determine_element_from_mass(atom.mass)
        elements.append(el)

    # PASS 1: Generate all ChimeraX residues ahead of time to avoid duplication/looping issues
    cx_residues = {}
    for res in u.residues:
        segid = getattr(res.segment, 'segid', '') 
        r = s.new_residue(res.resname, segid, res.resid)
        cx_residues[res.resindex] = r

    # PASS 2: Iterate flatly over u.atoms to ensure 1:1 mapping with the numpy coordinate array
    mda_to_cx = {}
    crd = tinyarray.array((0.0, 0.0, 0.0))
    
    for atom in u.atoms:
        sn = atom.id + 1 if hasattr(atom, 'id') else atom.index + 1
        el = elements[atom.index]
        r = cx_residues[atom.resindex]
        
        a = add_atom(name=atom.name, element=el, residue=r, loc=crd, serial_number=sn)
        mda_to_cx[atom.index] = a

    # PASS 3: Connect the bonds using the mapped indices
    if hasattr(u, 'bonds') and len(u.bonds) > 0:
        bonds_indices = u.bonds.to_indices()
        for i1, i2 in bonds_indices:
            try:
                add_bond(mda_to_cx[i1], mda_to_cx[i2])
            except KeyError:
                pass 

    return s
