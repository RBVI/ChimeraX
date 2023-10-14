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

def read_gro(session, stream, file_name, *, auto_style=True):
    structures = []
    try:
        lines = [line for line in stream]
        parse_gro(session, file_name, lines, structures, auto_style)
    except BaseException:
        for s in structures:
            s.delete()
        raise
    finally:
        stream.close()

    return structures, ""

from chimerax.atomic import AtomicStructure, Element

def parse_gro(session, file_name, lines, structures, auto_style):
    from chimerax.core.errors import UserError
    state = "init"
    anums = {}
    s = None
    for line in lines:
        line = line.rstrip()
        if line.startswith("#"):
            continue
        if state == "init":
            state = "post line 1"
            if ", t=" in line:
                mol_name = line[:line.index(", t=")]
            else:
                mol_name = line
            from chimerax.atomic.structure import is_informative_name
            name = mol_name if is_informative_name(mol_name) else file_name
            s = AtomicStructure(session, name=name, auto_style=auto_style)
            structures.append(s)
            continue
        if state == "post line 1":
            state = "atoms"
            num_atoms = int(line.strip())
            cur_res_num = None
            continue
        if not line: continue

        try:
            res_num = int(line[:5].strip())
            res_name = line[5:10].strip()
            atom_name = line[10:15].strip()
            atom_num = int(line[15:20].strip())
            x = float(line[20:28].strip()) * 10.0
            y = float(line[28:36].strip()) * 10.0
            z = float(line[36:44].strip()) * 10.0
        except ValueError:
            raise UserError("Atom line of gro file %s does not conform to format.\n"
                "Line: '%s'" % (file_name, line))
        if cur_res_num != res_num:
            r = s.new_residue(res_name, " ", res_num)
            cur_res_num = res_num
        if atom_name[0].upper() in "COPSHN" or len(atom_name) == 1:
            element = Element.get_element(atom_name[0].upper())
        else:
            twoLetter = atom_name[0].upper() + atom_name[1].lower()
            if twoLetter in Element.names:
                element = Element.get_element(twoLetter)
            else:
                element = Element.get_element(atom_name[0].upper())
        if element.number == 0:
            raise UserError("Cannot guess atomic element from atom name in file %s.\n"
                "Line: %s" % (file_name, line))
        anum = anums.get(element.name, 0) + 1
        anums[element.name] = anum
        a = s.new_atom(atom_name, element)
        r.add_atom(a)
        a.coord = (x, y, z)
        a.serial_number = atom_num
        if len(s.atoms) == num_atoms:
            break
    if s is None:
        raise UserError("'%s' has no non-comment lines!" % file_name)
    s.connect_structure()
