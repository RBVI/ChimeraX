# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

"""
sdf: MDL SDF/MOL format support
=======================

Read SDF files
"""

def read_sdf(session, stream, file_name):

    path = stream.name if hasattr(stream, 'name') else None

    structures = []
    nonblank = False
    state = "init"
    from chimerax.core.errors import UserError
    from chimerax.atomic.struct_edit import add_atom
    from chimerax.atomic import AtomicStructure, Element, Bond, Atom, AtomicStructure
    from numpy import array
    Bond.register_attr(session, "order", "SDF format", default_value=1.0, attr_type=float)
    Atom.register_attr(session, "charge", "SDF format", default_value=0.0, attr_type=float)
    AtomicStructure.register_attr(session, "charge_model", "SDF format", attr_type=str)
    try:
        for l in stream:
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
                s = AtomicStructure(session, name=name)
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
                    s.delete()
                    raise UserError("Atom line of MOL/SDF file '%s' is not x y z element...: '%s'"
                        % (file_name, l))
                element = Element.get_element(elem)
                if element.number == 0:
                    # lone pair of somesuch
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
                        % (file_name, 1))
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
            elif state == "data":
                if line == "$$$$":
                    nonblank = False
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
                        if not indexed_charges and len(atoms) != len(data):
                            raise UserError("Number of charges (%d) in %s data not equal to number of atoms"
                                " (%d)" % (len(data), orig_data_name, len(atoms)))
                        if indexed_charges:
                            for a in atoms:
                                # charge defaults to 0.0, so don't need to set non-indexed
                                for index, charge in data:
                                    atoms[index].charge = charge
                        else:
                            for a, charge in zip(atoms, data):
                                a.charge = charge
                        if "mmff94" in data_name:
                            s.charge_model = "MMFF94"
                        reading_data = None
                elif reading_data == "cid":
                    data_item = line.strip()
                    if data_item:
                        try:
                            cid = int(data_item)
                        except ValueError:
                            raise UserError("PubChem CID (%s) is %s data is not an integer" % (data_item,
                                orid_data_name))
                        s.prefix_html_title = False
                        s.get_html_title = lambda *args, cid=cid: 'PubChem entry <a href="https://pubchem.ncbi.nlm.nih.gov/compound/%d">%d</a>' % (cid, cid)
                        s.has_formatted_metadata = lambda *args: False
                        reading_data = None
                elif line.startswith('>'):
                    try:
                        lp = line.index('<')
                        rp = line[lp+1:].index('>') + lp + 1
                    except (IndexError, ValueError):
                        continue
                    orig_data_name = line[lp+1:rp]
                    data_name = orig_data_name.lower()
                    if data_name.endswith("charges") and "partial" in data_name:
                        reading_data = "charges"
                        indexed_charges = False
                        data = []
                    elif data_name == "pubchem_compound_cid":
                        reading_data = "cid"
    except:
        for s in structures:
            s.delete()
        raise
    finally:
        stream.close()

    if nonblank and state not in ["data", "init"]:
        if structures:
            session.logger.warning("Extraneous text after final $$$$ in MOL/SDF file '%s'" % file_name)
        else:
            raise UserError("Unexpected end of file (parser state: %s) in MOL/SDF file '%s'"
                % (state, file_name))

    return structures, ""
