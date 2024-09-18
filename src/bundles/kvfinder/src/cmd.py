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

def cmd_kvfinder(session, structures=None, *, extent=None, origin=None, probe_in=1.4, probe_out=4.0,
        removal_distance=2.4, step=0.6, surface='SES', volume_cutoff=5.0):
    if [origin, extent].count(None) == 1:
        raise UserError("Must specify both 'origin' and 'extent' or neither")
    from chimerax.atomic import all_atomic_structures, Structure
    if structures is None:
        structures = all_atomic_structures(session)
    if origin is not None:
        raise NotImplementedError("origin/extent keywords not yet implmented")
    from .prep import prep_input
    from chimerax.atomic.struct_edit import add_atom
    import numpy
    return_values = []
    for s in structures:
        insert_codes = s.residues.insertion_codes
        if len(insert_codes[insert_codes != '']) > 0:
            session.logger.warning("%s contains residue insertion codes; KVFinder may not work correctly"
                % s)
        struct_input, vertices = prep_input(s, origin, extent, probe_in, probe_out, step)
        nx, ny, nz = pyKVFinder.grid._get_dimensions(vertices, step)
        sincos = pyKVFinder.grid._get_sincos(vertices)
        num_cavities, cavity_matrix = pyKVFinder.detect(struct_input, vertices, step, probe_in, probe_out,
            removal_distance, volume_cutoff, None, 5.0, origin is not None, surface, None, False)
        session.logger.info("%d cavities found for %s" % (num_cavities, s))
        if num_cavities == 0:
            return_values.append((s, num_cavities, cavity_matrix, None))
            continue
        from chimerax.core.models import Model
        cavity_group = Model("cavities", session)
        s.add([cavity_group])
        return_values.append((s, num_cavities, cavity_matrix, cavity_group))
        model_lookup = {}
        used_colors = [[c/255 for c in s.overall_color], (0,0,0), (1,1,1)]
        from chimerax.core.colors import distinguish_from
        for i in range(num_cavities):
            cav_s = Structure(session, name="cavity %d" % (i+1), auto_style=False, log_info=False)
            r = cav_s.new_residue("CAV", "cavity", 1)
            cavity_group.add([cav_s])
            rgb = distinguish_from(used_colors, num_candidates=5, seed=71428)
            used_colors.append(rgb)
            model_lookup[i+2] = (cav_s, r, rgb + (1.0,))
        origin, *args = vertices
        assert (nx, ny, nz) == cavity_matrix.shape
        for xi in range(nx):
            x = origin[0] + xi * step
            for yi in range(ny):
                y = origin[1] + yi * step
                for zi in range(nz):
                    val = cavity_matrix[xi][yi][zi]
                    if val < 2:
                        continue
                    z = origin[2] + zi * step
                    cav_s, r, rgba = model_lookup[val]
                    a = add_atom("Z%d" % cav_s.num_atoms, "He", r, numpy.array((x,y,z)))
                    a.radius = 0.1
        for cav_s, r, rgba in model_lookup.values():
            cav_s.overall_color = [255.0 * c for c in rgba]

    return return_values

def register_command(command_name, logger):
    from chimerax.core.commands import CmdDesc, register, Or, EmptyArg, Float3Arg, FloatArg, EnumOf
    from chimerax.atomic import AtomicStructuresArg
    kw = {
        'required': [('structures', Or(AtomicStructuresArg, EmptyArg))],
        'keyword': [
            ('extent', Or(Float3Arg, FloatArg)),
            ('origin', Float3Arg),
            ('probe_in', FloatArg),
            ('probe_out', FloatArg),
            ('removal_distance', FloatArg),
            ('step', FloatArg),
            ('surface', EnumOf(['SAS', 'SES'])),
            ('volume_cutoff', FloatArg),
        ]
    }
    register(command_name, CmdDesc(**kw, synopsis="Find pockets"), cmd_kvfinder, logger=logger)
