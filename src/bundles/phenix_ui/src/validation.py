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

def read_probe_dots(session, data, file_name, structure_model, **kw):
    from chimerax.core.errors import UserError
    from chimerax.core.colors import Color
    pale_green = Color("pale green")
    cornflower_blue = Color("cornflower blue")
    sky_blue = Color("sky blue")
    aquamarine = Color("aquamarine")
    green = Color("green")
    yellow_green = Color("yellow green")
    yellow = Color("yellow")
    orange = Color("orange")
    red = Color("red")
    orange_red = Color("orange red")

    # set up a search tree so we can try to match probe identifiers to residues
    # (probe identifiers may be for non-existent hydrogen atoms)
    from chimerax.atom_search import AtomSearchTree
    search_tree = AtomSearchTree(structure_model.atoms, scene_coords=False)
    seen_probe_residues = set()
    dot_data = []
    line_num = 0
    for line in data:
        line_num += 1
        fields = line.split(':')
        if len(fields) != 19:
            raise UserError("Unexpected number of colon-separated fields on line %d of %s."
                "  Expected 19 and got %s." % (line_num, file_name, len(fields)))
        markup_type = fields[2]
        node_gap = float(fields[6])
        spike_xyz = [float(f) for f in fields[7:10]]
        node_xyz = [float(f) for f in fields[-5:-2]]
        probe_atoms = tuple(fields[3:5])
        #TODO: use a fake attribute (like Chain.identity to allow atoms to match multiple probe identifiers
        for probe_atom in probe_atoms:
            probe_res = probe_atom[:10]
            if probe_res not in seen_probe_residues:
                if not match_probe_to_cx(search_tree, probe_res, node_xyz):
                    session.logger.warning("Could not identify residue"
                        " corresponding to Probe identifier '%s'" % probe_res)
                seen_probe_residues.add(probe_res)
        if markup_type == "hb":
            color = pale_green
        elif node_gap > 0.35:
            color = cornflower_blue
        elif node_gap > 0.25:
            color = sky_blue
        elif node_gap > 0.15:
            color = aquamarine
        elif node_gap > 0.0:
            color = green
        elif node_gap > -0.1:
            color = yellow_green
        elif node_gap > -0.2:
            color = yellow
        elif node_gap > -0.3:
            color = orange
        elif node_gap > -0.4:
            color = red
        else:
            color = orange_red
        dot_data.append((node_xyz, (spike_xyz if node_gap > -0.3 else None), color))

def match_probe_to_cx(tree, probe, node_xyz):
    chain = probe[:2].strip()
    h36_res_num = probe[2:6].strip()
    if h36_res_num[0].isdigit():
        res_num = int(h36_res_num)
    else:
        res_num = 0
        for c in h36_res_num:
            resnum = res_num * 36
            if c.isdigit():
                val = int(c)
            else:
                val = 10 + ord(c) - ord('A')
            resnum += val
    insert_code = probe[6].strip()
    res_name = probe[7:10].strip()
    #atom_name = probe[11:15].strip()
    #alt_loc = probe[15]
    for atom in tree.search(node_xyz, 5.0):
        r = atom.residue
        if r.chain_id != chain:
            continue
        if r.number != res_num:
            continue
        if insert_code != r.insertion_code:
            continue
        if res_name != r.name:
            continue
        r.probe_id = probe[:10]
        return True
    return False

