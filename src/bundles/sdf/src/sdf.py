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

"""
sdf: MDL SDF/MOL format support
=======================

Read SDF files
"""

from chimerax.core.errors import UserError
from chimerax.atomic.struct_edit import add_atom
from chimerax.atomic import AtomicStructure, Element, Bond, Atom, AtomicStructure
from numpy import array

def read_sdf(session, stream, file_name, *, auto_style=True):

    structures = []
    Bond.register_attr(session, "order", "SDF format", attr_type=float)
    Atom.register_attr(session, "charge", "SDF format", attr_type=float)
    AtomicStructure.register_attr(session, "charge_model", "SDF format", attr_type=str)
    try:
        lines = [line for line in stream]
        if lines[3].strip().endswith("V3000"):
            parse_v3000(session, file_name, lines, structures, auto_style)
        else:
            parse_v2000(session, file_name, lines, structures, auto_style)
    except BaseException:
        for s in structures:
            s.delete()
        raise
    finally:
        stream.close()

    return structures, ""

def parse_v2000(session, file_name, lines, structures, auto_style):
    nonblank = False
    state = "init"
    for l in lines:
        line = l.strip()
        nonblank = nonblank or line
        if state == "init":
            state = "post header 1"
            mol_name = line
        elif state == "post header 1":
            state = "post header 2"
        elif state == "post header 2":
            state = "counts"
        elif state == "counts":
            if not line:
                break
            state = "atoms"
            serial = 1
            anums = {}
            atoms = []
            try:
                num_atoms = int(l[:3].strip())
                num_bonds = int(l[3:6].strip())
            except ValueError:
                raise UserError("Atom/bond counts line of MOL/SDF file '%s' is botched" % file_name)
            from chimerax.atomic.structure import is_informative_name
            name = mol_name if is_informative_name(mol_name) else file_name
            s = AtomicStructure(session, name=name, auto_style=auto_style)
            structures.append(s)
            r = s.new_residue("UNL", " ", 1)
        elif state == "atoms":
            num_atoms -= 1
            if num_atoms == 0:
                if num_bonds:
                    state = "bonds"
                else:
                    state = "properties"
            try:
                x = float(l[:10].strip())
                y = float(l[10:20].strip())
                z = float(l[20:30].strip())
                elem = l[31:34].strip()
            except ValueError:
                raise UserError("Atom line of MOL/SDF file '%s' is not x y z element...: '%s'"
                    % (file_name, line))
            element = Element.get_element(elem)
            if element.number == 0:
                # lone pair or somesuch
                atoms.append(None)
                continue
            anum = anums.get(element.name, 0) + 1
            anums[element.name] = anum
            a = add_atom("%s%d" % (element.name, anum), element, r, array([x,y,z]), serial_number=serial)
            serial += 1
            atoms.append(a)
        elif state == "bonds":
            num_bonds -= 1
            if num_bonds == 0:
                state = "properties"
            try:
                a1_index = int(l[:3].strip())
                a2_index = int(l[3:6].strip())
                order = float(l[6:9].strip())
            except ValueError:
                raise UserError("Bond line of MOL/SDF file '%s' is not a1 a2 order...: '%s'"
                    % (file_name, line))
            a1 = atoms[a1_index-1]
            a2 = atoms[a2_index-1]
            if not a1 or not a2:
                continue
            s.new_bond(a1, a2).order = order
        elif state == "properties":
            if not s.atoms:
                raise UserError("No atoms found for compound '%s' in MOL/SDF file '%s'" % (name, file_name))
            if line.split() == ["M", "END"]:
                state = "data"
                reading_data = None
                charge_data = []
                indexed_charges = False
                data_name = orig_data_name = None
        elif state == "data":
            state, reading_data, indexed_charges, data_name, orig_data_name = read_data_line(s, state,
                reading_data, line, atoms, charge_data, indexed_charges, data_name, orig_data_name)
            if state == "init":
                nonblank = False
    if nonblank and state not in ["data", "init"]:
        if structures:
            session.logger.warning("Extraneous text after final $$$$ in MOL/SDF file '%s'" % file_name)
        else:
            raise UserError("Unexpected end of file (parser state: %s) in MOL/SDF file '%s'"
                % (state, file_name))

def parse_v3000(session, file_name, lines, structures, auto_style):
    lines = [l.strip() for l in lines]
    default_mol_name = lines[0]
    blocks = []
    for ln4, line in enumerate(lines[4:]):
        line_num = ln4 + 4
        if line.startswith("M  V30 "):
            fields = line[7:].split()
            if not fields:
                continue
            if fields[0] == "BEGIN":
                if len(fields) == 1:
                    raise UserError("SDF V3000 file %s, line %s blank after '%s'"
                        % (file_name, line_num, line))
                block_type = fields[1]
                if block_type == "CTAB":
                    if blocks:
                        raise UserError("SDF V3000 file %s has nested CTAB block at line %d"
                            % (file_name, line_num))
                    if len(fields) > 2:
                        mol_name = " ".join(fields[2:])
                    else:
                        mol_name = default_mol_name
                    from chimerax.atomic.structure import is_informative_name
                    name = mol_name if is_informative_name(mol_name) else file_name
                    s = AtomicStructure(session, name=name, auto_style=auto_style)
                    atom_info = {}
                    atoms = []
                    anums = {}
                    serial = 1
                    structures.append(s)
                    r = s.new_residue("UNL", " ", 1)
                elif block_type in ["ATOM", "BOND"] and (not blocks or blocks[0] != "CTAB"):
                    raise UserError("SDF V3000 file %s, line %d: %s block not within CTAB block"
                        % (file_name, line_num, block_type))
                blocks.append(block_type)
            elif fields[0] == "END":
                if len(fields) == 1:
                    raise UserError("SDF V3000 file %s, line %d blank after '%s'"
                        % (file_name, line_num, line))
                block_type = fields[1]
                if not blocks or block_type != blocks[-1]:
                    raise UserError("SDF V3000 file %s, line %d: END doesn't match BEGIN"
                        % (file_name, line_num))
                blocks.pop()
                if block_type == "CTAB" and not blocks:
                    state = "data"
                    reading_data = None
                    charge_data = []
                    indexed_charges = False
                    data_name = orig_data_name = None
            elif blocks:
                if blocks[-1] == "ATOM":
                    try:
                        index, element, x, y, z, *ignored = fields
                    except ValueError:
                        raise UserError("SDF V3000 file %s, line %d: need at least 5 fields"
                            " (index, element, x, y, z) for ATOM data line" % (file_name, line_num))
                    try:
                        index = int(index)
                        x = float(x)
                        y = float(y)
                        z = float(z)
                    except ValueError:
                        raise UserError("SDF V3000 file %s, line %d: index must be integer and x, y, z"
                            " must be floating point for ATOM data line" % (file_name, line_num))
                    if index in atom_info:
                        raise UserError("SDF V3000 file %s, line %d: indices must be unique"
                            " across ATOM data lines" % (file_name, line_num))
                    element = Element.get_element(element)
                    if element.number == 0:
                        # lone pair or somesuch
                        a = None
                    else:
                        anum = anums.get(element.name, 0) + 1
                        anums[element.name] = anum
                        a = add_atom("%s%d" % (element.name, anum), element, r, array([x,y,z]),
                            serial_number=serial)
                        serial += 1
                    atom_info[index] = a
                elif blocks[-1] == "BOND":
                    try:
                        index, order, i1, i2, *ignored = fields
                    except ValueError:
                        raise UserError("SDF V3000 file %s, line %d: need at least 4 fields"
                            " (index, bond type, x, y, z) for BOND data line" % (file_name, line_num))
                    try:
                        i1 = int(i1)
                        i2 = int(i2)
                    except ValueError:
                        raise UserError("SDF V3000 file %s, line %d: indices must be integers"
                            " for BOND data line" % (file_name, line_num))
                    if i1 not in atom_info or i2 not in atom_info:
                        raise UserError("SDF V3000 file %s, line %d: indices not in ATOM block"
                            " for BOND data line" % (file_name, line_num))
                    a1, a2 = atom_info[i1], atom_info[i2]
                    if a1 and a2 and a1 not in a2.neighbors:
                        s.new_bond(a1, a2)
        elif structures:
            if state != "init":
                state, reading_data, indexed_charges, data_name, orig_data_name = read_data_line(s,
                    state, reading_data, line, atoms, charge_data, indexed_charges, data_name,
                    orig_data_name)
    if blocks:
        raise UserError("SDF V3000 file %s: no END found for %s block" % (file_name, blocks[-1]))

def read_data_line(s, state, reading_data, line, atoms, data, indexed_charges, data_name, orig_data_name):
    if line == "$$$$":
        state = "init"
    elif reading_data == "charges":
        data_item = line.strip()
        if data_item:
            try:
                data.append(float(data_item))
            except ValueError:
                try:
                    index, charge = data_item.split()
                    index = int(index) - 1
                    charge = float(charge)
                except ValueError:
                    raise UserError("Charge data (%s) in %s data is not either a floating-point"
                        " number or an atom index and a floating-point number" % (data_item,
                        orig_data_name))
                else:
                    if not indexed_charges:
                        # for indexed charges, the first thing is a count
                        data.pop()
                        indexed_charges = True
                    data.append((index, charge))
        else:
            reading_data = None
            # single value of '0' indicates that charges are not being provided
            if not (len(data) == 1 and len(atoms) != 1 and data[0] == 0.0):
                if not indexed_charges and len(atoms) != len(data):
                    raise UserError("Number of charges (%d) in %s data not equal to number of atoms"
                        " (%d)" % (len(data), orig_data_name, len(atoms)))
                if indexed_charges:
                    for a in atoms:
                        # charge defaults to 0.0, so don't need to set non-indexed
                        for index, charge in data:
                            atoms[index].charge = charge
                else:
                    indices = list(atoms.keys())
                    indices.sort
                    for a, charge in zip([atoms[i] for i in indices], data):
                        a.charge = charge
                if "mmff94" in data_name:
                    s.charge_model = "MMFF94"
    elif reading_data == "cid":
        data_item = line.strip()
        if data_item:
            try:
                cid = int(data_item)
            except ValueError:
                raise UserError("PubChem CID (%s) is %s data is not an integer" % (data_item,
                    orid_data_name))
            s.name = "pubchem:%d" % cid
            s.prefix_html_title = False
            s.get_html_title = lambda *args, cid=cid: 'PubChem entry <a href="https://pubchem.ncbi.nlm.nih.gov/compound/%d">%d</a>' % (cid, cid)
            s.has_formatted_metadata = lambda *args: False
            reading_data = None
    elif line.startswith('>'):
        try:
            lp = line.index('<')
            rp = line[lp+1:].index('>') + lp + 1
        except (IndexError, ValueError):
            pass
        else:
            orig_data_name = line[lp+1:rp]
            data_name = orig_data_name.lower()
            if data_name.endswith("charges") and "partial" in data_name:
                reading_data = "charges"
            elif data_name == "pubchem_compound_cid":
                reading_data = "cid"
    return state, reading_data, indexed_charges, data_name, orig_data_name
