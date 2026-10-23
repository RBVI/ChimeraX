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
    pale_green = Color("pale green").uint8x4()
    cornflower_blue = Color("cornflower blue").uint8x4()
    sky_blue = Color("sky blue").uint8x4()
    aquamarine = Color("aquamarine").uint8x4()
    green = Color("green").uint8x4()
    yellow_green = Color("yellow green").uint8x4()
    yellow = Color("yellow").uint8x4()
    orange = Color("orange").uint8x4()
    red = Color("red").uint8x4()
    orange_red = Color("orange red").uint8x4()

    from numpy import array, concatenate, empty, float32, int32, array_equal
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
        spike_xyz = array([float(f) for f in fields[7:10]])
        node_xyz = array([float(f) for f in fields[-5:-2]])
        #probe_atoms = tuple(fields[3:5])
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


    from chimerax.core.models import Model
    probe_model = Model("Probe markups", session)
    structure_model.add([probe_model])
    drawing = probe_model.new_drawing('Probe markup')
    # 'shape' command defaults to 3810 triangles for a sphere of radius 1; scale down for the radius
    # we're using
    radius = 0.05
    from chimerax.geometry import sphere, translation
    from chimerax.surface import calculate_vertex_normals
    from chimerax.shape.shape import cylinder_divisions, cylinder_geometry
    # number of triangles has to be even
    varray1, tarray1 = sphere.sphere_triangulation(2 * round(radius * 3810) // 2)
    varray1 *= radius
    varrays = []
    narrays = []
    tarrays = []
    rgba8s = []
    num_vertices = len(varray1)
    spk_radius = radius / 4
    for i, dot_info in enumerate(dot_data):
        node_xyz, spike_xyz, rgba8 = dot_info
        varray = translation(node_xyz).transform_points(varray1)
        narray = calculate_vertex_normals(varray, tarray1)
        varrays.append(varray)
        narrays.append(narray)
        tarrays.append(tarray1 + i * num_vertices)
        rgba8s.extend([rgba8] * num_vertices)
        if array_equal(node_xyz, spike_xyz):
            # no spike
            continue
        #TODO
    num_all_vertices = len(varray1) * len(dot_data)
    num_all_triangles = len(tarray1) * len(dot_data)
    vertices = empty((num_all_vertices, 3), dtype=float32)
    normals = empty((num_all_vertices, 3), dtype=float32)
    triangles = empty((num_all_triangles, 3), dtype=int32)

    probe_model.set_geometry(concatenate(varrays, out=vertices), concatenate(narrays, out=normals),
        concatenate(tarrays, out=triangles))
    probe_model.vertex_colors = array(rgba8s)

    # Insert fake attribute (ala Chain.identity) to allow matching to probe identifiers
    # (particulary since probe identifiers can be for non-existent hydrogens).
    # Can only add properties to classes (not instances), so...
    def probe_id(self):
        return ProbeTester(self)
    from chimerax.atomic import Atom
    Atom.probe_id = property(probe_id)

class ProbeTester():
    def __init__(self, atom):
        self.atom = atom

    def __eq__(self, probe_id):
        return self._probe_id_matches(probe_id)

    def __ne__(self, probe_id):
        return not self._probe_id_matches(probe_id)

    def __str__(self):
        # raise TypeError so that attribute-testing code can catch it and do the right thing
        raise TypeError("fake attribute")

    def _probe_id_matches(self, probe_id):
        try:
            # testing mechanism downcases strings with "=" test, so...
            probe_id = probe_id.upper()
            chain = probe_id[:2].strip()
            h36_res_num = probe_id[2:6].strip()
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
            insert_code = probe_id[6].strip()
            res_name = probe_id[7:10].strip()
            atom_name = probe_id[11:15].strip()
            #alt_loc = probe_id[15]
        except:
            return False
        r = self.atom.residue
        if r.chain_id != chain:
            return False
        if r.number != res_num:
            return False
        if insert_code != r.insertion_code:
            return False
        if res_name != r.name:
            return False
        # atom name matching is a little tricky, since if matches our name exactly that's easy,
        # but if it's a hydrogen an we're a heavy atom it might still match us since Probe
        # adds non-existent hydrogens
        if atom_name == self.atom.name:
            return True
        if self.atom.element.name == 'H' or atom_name[0] != 'H':
            return False
        for a in r.atoms:
            if a.name == atom_name:
                # matches a different atom
                return False
        my_prefix = self.atom.element.name.upper()
        if not self.atom.name.startswith(my_prefix):
            return False
        a_test = self.atom.name[len(my_prefix):]
        h_test = atom_name[1:]
        if not h_test:
            # the 'H' attached to N
            return self.atom.name == 'N'
        if h_test == a_test:
            return True
        if h_test[-1].isdigit():
            h_test = h_test[:-1]
        return a_test == h_test

