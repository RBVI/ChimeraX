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

from chimerax.core.errors import LimitationError, UserError
import pyKVFinder

def cmd_kvfinder(session, structures=None, *, box_extent=None, box_origin=None, probe_in=1.4, probe_out=4.0,
        removal_distance=2.4, show_tool=True, grid_spacing=0.6, surface_type='SES', volume_cutoff=5.0,
        replace=True):
    if [box_origin, box_extent].count(None) == 1:
        raise UserError("Must specify both 'boxOrigin' and 'boxExtent' or neither")
    from chimerax.atomic import all_atomic_structures, Structure, Atom, Residues
    for attr_name in ["area", "volume", "max_depth", "average_depth"]:
        Structure.register_attr(session, "kvfinder_" + attr_name, "KVFinder", attr_type=float)
    Atom.register_attr(session, "kvfinder_depth", "KVFinder", attr_type=float)
    if structures is None:
        structures = all_atomic_structures(session)
    show_tool = show_tool and session.ui.is_gui
    from .prep import prep_input
    from chimerax.atomic.struct_edit import add_atom
    import numpy
    return_values = []
    cavity_group_name = "cavities"
    for s in structures:
        session.logger.status("Find Cavities for %s: preparing KVFinder input" % s)
        insert_codes = s.residues.insertion_codes
        if len(insert_codes[insert_codes != '']) > 0:
            session.logger.warning("%s contains residue insertion codes; KVFinder may not work correctly"
                % s)
        struct_input, vertices = prep_input(s, box_origin, box_extent, probe_in, probe_out, grid_spacing)
        session.logger.status("Find Cavities for %s: getting grid dimensions" % s)
        nx, ny, nz = pyKVFinder.grid._get_dimensions(vertices, grid_spacing)
        sincos = pyKVFinder.grid._get_sincos(vertices)
        session.logger.status("Find Cavities for %s: finding cavities" % s)
        num_cavities, cavity_matrix = pyKVFinder.detect(struct_input, vertices, grid_spacing,
            probe_in, probe_out, removal_distance, volume_cutoff, None, 5.0, box_origin is not None,
            surface_type, None, False)
        for ignore_backbone in [True, False]:
            contact_res_info = pyKVFinder.grid.constitutional(cavity_matrix, struct_input, vertices,
                grid_spacing, probe_in, ignore_backbone)
            processed_res_info = {}
            for cav_id, res_info in contact_res_info.items():
                processed_res_info[cav_id] = Residues([s.find_residue(cid, int(pos))
                    for pos, cid, name in res_info])
            if ignore_backbone:
                non_backbone_contacting = processed_res_info
            else:
                contacting = processed_res_info

        placement = None
        if replace:
            closures = []
            for child in s.child_models():
                if child.name == cavity_group_name:
                    closures.append(child)
            if closures:
                if show_tool:
                    from .tool import KVFinderResultsDialog
                    for tool in session.tools:
                        if isinstance(tool, KVFinderResultsDialog):
                            if tool.cavity_group in closures:
                                placement = None if tool.tool_window.floating else 'side'
                                break
                session.models.close(closures)
        session.logger.info("%d cavities found for %s" % (num_cavities, s))
        if num_cavities == 0:
            return_values.append((s, num_cavities, cavity_matrix, None))
            continue
        session.logger.status("Find Cavities for %s: determining cavities' surface/volume" % s)
        surface_grid, k_volume, k_area = pyKVFinder.spatial(cavity_matrix, step=grid_spacing)
        session.logger.status("Find Cavities for %s: finding cavity depths" % s)
        depths, max_depth, avg_depth = pyKVFinder.depth(cavity_matrix, step=grid_spacing)
        session.logger.status("Find Cavities for %s: creating cavity models" % s)
        from chimerax.core.models import Model
        cavity_group = Model(cavity_group_name, session)
        s.add([cavity_group])
        return_values.append((s, num_cavities, cavity_matrix, cavity_group))
        model_lookup = {}
        used_colors = [(0,0,0), (1,1,1)]
        overall_color = s.overall_color
        if overall_color is not None:
            used_colors.append([c/255 for c in overall_color])
        from chimerax.core.colors import distinguish_from
        for i in range(num_cavities):
            cav_s = Structure(session, name="cavity %d" % (i+1), auto_style=False, log_info=False)
            r = cav_s.new_residue("CAV", "cavity", 1)
            cavity_group.add([cav_s])
            rgb = distinguish_from(used_colors, num_candidates=5, seed=71428)
            used_colors.append(rgb)
            model_lookup[i+2] = (cav_s, r, rgb + (1.0,))
            # map 'KXX"-indexed volume/area to usable indices
            k_index = pyKVFinder.grid._get_cavity_name(i)
            cav_s.kvfinder_area = k_area[k_index]
            cav_s.kvfinder_volume = k_volume[k_index]
            cav_s.kvfinder_max_depth = max_depth[k_index]
            cav_s.kvfinder_average_depth = avg_depth[k_index]
            contacting[cav_s] = contacting[k_index]
            non_backbone_contacting[cav_s] = non_backbone_contacting[k_index]
            del contacting[k_index]
            del non_backbone_contacting[k_index]
        origin, *args = vertices
        assert (nx, ny, nz) == cavity_matrix.shape
        # Using the explicit triple loop instead of more numpy-like code, because AFAICT the
        # numpy code would have to process the matrix 'num_cavities' times, so therefore winds up
        # being slower than non-numpy code processing it once.
        # Example numpy code:
        #    for val in range(2, 2+num_cavities):
        #        cav_s, r, rgba = model_lookup[val]
        #        xyzs = numpy.argwhere(cavity_matrix == val) * grid_spacing + origin
        #        for xyz in xyzs:
        #            a = add_atom("Z%d" % cav_s.num_atoms, "He", r, xyz))
        #            a.radius = 0.1
        session.logger.status("Find Cavities for %s: filling in cavity models" % s)
        cavity_iter = cavity_matrix.flat
        depth_iter = depths.flat
        for xi in range(nx):
            x = origin[0] + xi * grid_spacing
            for yi in range(ny):
                y = origin[1] + yi * grid_spacing
                for zi in range(nz):
                    #val = cavity_matrix[xi][yi][zi]
                    val = int(next(cavity_iter))
                    depth = float(next(depth_iter))
                    if val < 2:
                        continue
                    z = origin[2] + zi * grid_spacing
                    cav_s, r, rgba = model_lookup[val]
                    a = add_atom("Z%d" % cav_s.num_atoms, "He", r, numpy.array((x,y,z)))
                    a.kvfinder_depth = depth
        for cav_s, r, rgba in model_lookup.values():
            cav_s.overall_color = [255.0 * c for c in rgba]
            cav_s.ball_scale = 0.25
            atoms = cav_s.atoms
            atoms.radii = grid_spacing / 2
            atoms.draw_modes = atoms.BALL_STYLE
        from chimerax.core.logger import html_table_params
        table_lines = [
            '<table %s>' % html_table_params,
            '  <thead>',
            '    <tr>',
            '      <th colspan="7">%s Cavities</th>' % s.name,
            '    </tr>',
            '    <tr>',
            '      <th>ID</th>',
            '      <th></th>',
            '      <th>Volume</th>',
            '      <th>Area</th>',
            '      <th>Points</th>',
            '      <th>Maximum<br>Depth</th>',
            '      <th>Average<br>Depth</th>',
            '    </tr>',
            '  </thead>',
            '  <tbody>',
        ]
        cavity_info = [(v[0], v[-1]) for v in model_lookup.values()]
        cavity_info.sort(key=lambda info: -info[0].kvfinder_volume)
        for cav_s, rgba in cavity_info:
            table_lines.extend([
            '    <tr>',
            '      <td style="text-align:center">%s</td>' % cav_s.id_string,
            '      <td style="background-color:rgb(%d, %d, %d)"></td>'
                        % tuple([round(255.0 * c) for c in rgba[:3]]),
            '      <td style="text-align:center">%g</td>' % cav_s.kvfinder_volume,
            '      <td style="text-align:center">%g</td>' % cav_s.kvfinder_area,
            '      <td style="text-align:center">%d</td>' % cav_s.num_atoms,
            '      <td style="text-align:center">%g</td>' % cav_s.kvfinder_max_depth,
            '      <td style="text-align:center">%g</td>' % cav_s.kvfinder_average_depth,
            '    </tr>',
            ])
        table_lines.extend([
            '  </tbody>',
            '</table>'
        ])
        session.logger.info('\n'.join(table_lines), is_html=True)
        if show_tool:
            from .tool import KVFinderResultsDialog
            KVFinderResultsDialog(session, "%s Cavities" % s.name, s, cavity_group,
                [ml[0] for ml in model_lookup.values()], probe_in, placement=placement,
                contacting_info=(contacting, non_backbone_contacting))
        session.logger.status("Find Cavities for %s: done" % s)

    return return_values

def register_command(command_name, logger):
    from chimerax.core.commands import CmdDesc, register, Or, EmptyArg, Float3Arg, FloatArg, EnumOf, BoolArg
    from chimerax.atomic import AtomicStructuresArg
    kw = {
        'required': [('structures', Or(AtomicStructuresArg, EmptyArg))],
        'keyword': [
            ('box_extent', Or(Float3Arg, FloatArg)),
            ('box_origin', Float3Arg),
            ('probe_in', FloatArg),
            ('probe_out', FloatArg),
            ('removal_distance', FloatArg),
            ('replace', BoolArg),
            ('show_tool', BoolArg),
            ('grid_spacing', FloatArg),
            ('surface_type', EnumOf(['SAS', 'SES'])),
            ('volume_cutoff', FloatArg),
        ]
    }
    register(command_name, CmdDesc(**kw, synopsis="Find pockets"), cmd_kvfinder, logger=logger)
