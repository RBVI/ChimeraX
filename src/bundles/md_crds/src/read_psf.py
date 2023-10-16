# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

from chimerax.atomic import AtomicStructure, Element

def determine_element_from_mass(mass, *, consider_hydrogens=True):
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

def read_psf(session, path, file_name, *, auto_style=True, coords=None, **kw):
    from chimerax.core.errors import UserError, CancelOperation
    import os
    if coords is None:
        if session.ui.is_gui and not session.in_script:
            from Qt.QtWidgets import QFileDialog
            coords, types = QFileDialog.getOpenFileName(caption="Specify coordinates file for PSF",
                directory=os.path.dirname(path))
            if not coords:
                raise CancelOperation("No coordinates file specified for PSF")
            session.logger.info("Coordinates file: %s" % coords)
        else:
            raise UserError("'coords' keyword with coordinate-file argument must be supplied")
    from chimerax.data_formats import NoFormatError
    try:
        data_fmt = session.data_formats.open_format_from_file_name(coords)
    except NoFormatError as e:
        raise UserError("Cannot determine format of coordinates file '%s' from suffix" % coords)

    from .dcd.MDToolsMarch97 import md
    try:
        mdt_mol = md.Molecule(psf=path)
        mdt_mol.buildstructure()
    except Exception as e:
        raise UserError("Problem reading/processing PSF file '%s': %s" % (path, e))

    s = AtomicStructure(session, name=os.path.basename(file_name), auto_style=auto_style)
    try:
        from chimerax.atomic.struct_edit import add_atom, add_bond
        import tinyarray
        crd = tinyarray.array((0.0, 0.0, 0.0))
        atom_index = 0
        res_index = 0
        res_order = {}
        for seg in mdt_mol.segments:
            for sres in seg.residues:
                r = s.new_residue(sres.name, " ", sres.id)
                res_order[r] = res_index
                res_index += 1
                n_atoms = len(sres.atoms)
                for i in range(atom_index, atom_index+n_atoms):
                    atom = mdt_mol.atoms[i]
                    add_atom(atom.name, determine_element_from_mass(atom.mass), r, crd)
                atom_index += n_atoms
        psf_index = { pa:i for i, pa in enumerate(mdt_mol.atoms) }
        atoms = s.atoms
        for i1, i2 in mdt_mol.bonds:
            add_bond(atoms[psf_index[i1]], atoms[psf_index[i2]])

        from chimerax.atomic import next_chain_id
        chain_id = 'A'
        res_groups = [grp.unique_residues for grp in s.bonded_groups()]
        multi_res_groups = [grp for grp in res_groups if len(grp) > 1]
        for grp in sorted(multi_res_groups, key=lambda grp: res_order[grp[0]]):
            for r in grp:
                r.chain_id = chain_id
            chain_id = next_chain_id(chain_id)

        # CHARMM QM simulations can put parts of of the same residue into different
        # parts of the PSF file; merge residues with identical identifiers and types
        # that are in the same bonded group
        for grp in multi_res_groups:
            res_lookup = {}
            for r in grp:
                key = (r.name, r.number)
                if key in res_lookup:
                    prev = res_lookup[key]
                    for a in r.atoms:
                        r.remove_atom(a)
                        prev.add_atom(a)
                    s.delete_residue(r)
                else:
                    res_lookup[key] = r

        from .read_coords import read_coords
        read_coords(session, coords, s, data_fmt.nicknames[0], replace=True, **kw)
    except BaseException:
        s.delete()
        raise

    return [s], ""
