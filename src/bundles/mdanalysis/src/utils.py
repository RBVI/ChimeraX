# vim: set expandtab shiftwidth=4 softtabstop=4:

from chimerax.atomic import Element

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
    
    session.logger.info(f"*** ok 3\n")

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

    from chimerax.atomic import next_chain_id
    current_chain_id = "A"
    
    session.logger.info(f"*** ok 4\n")

    
    for seg in u.segments:
        # If segment has a segid, try to use it as chain ID if it's short, else generate
        segid = seg.segid.strip()
        if len(segid) == 1 and segid.isalnum():
            chain_id = segid
        else:
            chain_id = current_chain_id
            current_chain_id = next_chain_id(current_chain_id)

        for res in seg.residues:
            # Create ChimeraX residue
            # res.resname, res.resid
            r = s.new_residue(res.resname, chain_id, res.resid)
            
            for atom in res.atoms:
                el = elements[atom.index] # atom.index is global 0-based index
                a = add_atom(atom.name, el, r, crd)
                a.serial_number = atom.id if hasattr(atom, 'id') else atom.index + 1
                mda_to_cx[atom.index] = a

    session.logger.info(f"*** ok 5\n")

    # Set coordinates for the initial frame
    if u.trajectory.n_frames > 0:
        pos = u.atoms.positions.astype(np.float64)
        s.atoms.coords = pos
        
    session.logger.info(f"*** ok 6\n")


    # Add bonds
    # MDA bonds are (atom1, atom2) tuples (or Bond objects)
    if hasattr(u, 'bonds') and len(u.bonds) > 0:
        # Convert bonds to index pairs to avoid object overhead loop
        bonds_indices = u.bonds.to_indices()
        for i1, i2 in bonds_indices:
            try:
                add_bond(mda_to_cx[i1], mda_to_cx[i2])
            except KeyError:
                pass # Should not happen if topology is consistent

    session.logger.info(f"*** ok 7\n")

    s.connect_structure()
    
    session.logger.info(f"*** ok 8\n")

    return s
