# vim: set expandtab shiftwidth=4 softtabstop=4:

from chimerax.atomic import Element
from chimerax.core.errors import UserError
import numpy as np
import MDAnalysis as mda
import os

def read_structure(session, path, file_name, format_name=None, *, auto_style=True, coords=None, **kw):
    """
    Unified structure reader using MDAnalysis.
    Handles PSF, GRO, LAMMPS Data, etc.
    """


    # If an external coordinate file is provided (e.g. PSF + DCD)
    if coords:
        coords_path, _ = prep_coords(session, coords, path, format_name)
        load_args = [path, coords_path]
    else:
        load_args = [path]

    try:
        # Map ChimeraX format names to MDA format keywords if necessary
        # MDA usually auto-detects, but we can be explicit
        fmt_map = {
            "psf": "PSF",
            "gro": "GRO",
            "data": "DATA", # LAMMPS
            "pdb": "PDB"
        }
        mda_fmt = fmt_map.get(format_name, None)
        
        # Determine format kwarg
        kwargs = {}
        if mda_fmt:
            kwargs['format'] = mda_fmt
            
        # LAMMPS Data specific: MDA needs atom_style usually, but defaults to 'full'
        # If it fails, we might need to retry, but let's assume standard for now.
            
        universe = mda.Universe(path, coords, topology_format='PSF', format='LAMMPSDUMP', dt=1.0, in_memory=False)
        
    except ImportError:
        raise UserError("MDAnalysis is not installed. Please run 'pip install MDAnalysis' to use this bundle.")
    except Exception as e:
        session.logger.warning(f"MDAnalysis failed to load {path}: {e}")
        # Detailed error for LAMMPS which is common
        if format_name == "data":
             raise UserError(f"Failed to read LAMMPS data file. MDAnalysis error: {e}\n"
                             "Ensure the file has a standard header and Masses/Atoms sections.")
        raise UserError(f"Could not read file {file_name}: {e}")

    # Convert to ChimeraX structure
    try:
        name = os.path.basename(file_name)
        model = universe_to_atomic_structure(session, universe, name, auto_style=auto_style)
    except Exception as e:
         raise UserError(f"Failed to convert MDAnalysis topology to ChimeraX structure: {e}")

    # If external coords were loaded via MDA, we have one frame. 
    # If the file itself has coords (GRO), we have one frame.
    # PSF has 0 frames unless linked with coords.
    
    msg = f"Imported {model.num_atoms} atoms, {len(universe.trajectory)} frames."
    if len(universe.trajectory) > 0:
        #msg += f" Loaded frame 0 from {u.trajectory.filename}."
        
        for timestep in universe.trajectory:
            #session.logger.info(f"*** frame {timestep.frame} |positions| {len(timestep.positions)}")
            #session.logger.info(f"*** #{timestep.frame} {timestep.positions[:3]}")
            model.add_coordset(id=timestep.frame, xyz=timestep.positions.astype(np.float64))


    return [model], msg

# vim: set expandtab shiftwidth=4 softtabstop=4:


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

    if consider_hydrogens:
        max_hyds = 6
    else:
        max_hyds = 0

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

def prep_coords(session, coords_file, input, format_name, *, file_type="coordinates"):
    """Helper to handle file dialogs if coords_file is missing."""
    from chimerax.core.errors import UserError, CancelOperation
    if coords_file is None:
        if session.ui.is_gui and not session.in_script:
            import os
            if isinstance(input, str):
                path = input
            elif hasattr(input, 'name'):
                path = os.path.dirname(os.path.realpath(input.name))
            else:
                path = os.getcwd()
                
            from Qt.QtWidgets import QFileDialog
            coords, types = QFileDialog.getOpenFileName(
                caption=f"Specify {file_type} file for {format_name}",
                directory=path, options=QFileDialog.DontUseNativeDialog)
            if not coords:
                raise CancelOperation(f"No coordinates file specified for {format_name}")
            session.logger.info("Coordinates file: %s" % coords)
        else:
            raise UserError("'coords' keyword with coordinate-file argument must be supplied")
    else:
        coords = coords_file
    
    # Try to determine format if not explicit, but return just path usually
    from chimerax.data_formats import NoFormatError
    try:
        data_fmt = session.data_formats.open_format_from_file_name(coords)
    except NoFormatError:
        data_fmt = None # Not critical if we know the format from caller
        
    return coords, data_fmt

def universe_to_atomic_structure(session, u, name, auto_style=True):
    """
    Converts an MDAnalysis Universe to a ChimeraX AtomicStructure.
    """
    from chimerax.atomic import AtomicStructure
    from chimerax.atomic.struct_edit import add_atom, add_bond
    import tinyarray
    import numpy as np

    s = AtomicStructure(session, name=name, auto_style=auto_style)
    
    # MDAnalysis uses 'segments' which map well to Chains in ChimeraX
    # If no segments, everything is effectively one chain
    
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

    # We map MDA atoms to ChimeraX atoms to rebuild bonds later
    # mda_index -> cx_atom
    # Note: MDA indices are 0-based
    
    # It is faster to iterate by segment -> residue -> atom
    
    # Track residues to avoid duplication if they are split in the file
    # (Though MDA usually handles split residues by index, ChimeraX needs unique residues)
    
    crd = tinyarray.array((0.0, 0.0, 0.0)) # Placeholder, coords set later
    mda_to_cx = {}
    
    res_index = 0
    res_order = {}
    
    sorted_segments = sorted(u.segments, key=lambda seg: seg.residues[0].atoms[0].index)
    
    session.logger.info(f"*** sorted_segments {sorted_segments}")
    
    for seg in sorted_segments:
        for res in seg.residues:
            # Create ChimeraX residue
            # res.resname, res.resid
            r = s.new_residue(res.resname, seg.segid, res.resid)
            res_order[r] = res_index
            res_index += 1
            for atom in res.atoms:
                sn = atom.id+1 if hasattr(atom, 'id') else atom.index + 1

                #session.logger.info(f"*** {seg.segid} {res.resname}{res.resid} index {atom.index} sn {sn} name {atom.name}")

                el = elements[atom.index] # atom.index is global 0-based index

                a = add_atom(name=atom.name, element=el, residue=r, loc=crd, serial_number=sn)
                mda_to_cx[atom.index] = a

    # Add bonds
    # MDA bonds are (atom1, atom2) tuples (or Bond objects)
    if hasattr(u, 'bonds') and len(u.bonds) > 0:
        # Convert bonds to index pairs to avoid object overhead loop
        bonds_indices = u.bonds.to_indices()

        for i1, i2 in bonds_indices:
            try:
                #session.logger.info(f"*** mda_to_cx[{i1}] {mda_to_cx[i1]} mda_to_cx[{i2}] {mda_to_cx[i2]}")
                add_bond(mda_to_cx[i1], mda_to_cx[i2])
            except KeyError:
                pass # Should not happen if topology is consistent
    
    return s
