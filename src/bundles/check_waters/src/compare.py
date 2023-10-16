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

def compare_waters(input_model, output_model, overlap_distance=2):
    '''
    Find how many output waters overlap input waters.
    '''
    # Get water residues in input and output models
    input_waters = _water_residues(input_model)
    output_waters = _water_residues(output_model)

    # Get water oxygen coodinates and see which overlap.
    from chimerax.atomic import Atoms
    ia = Atoms([r.find_atom('O') for r in input_waters])
    ixyz = ia.scene_coords
    oa = Atoms([r.find_atom('O') for r in output_waters])
    oxyz = oa.scene_coords
    from chimerax.geometry import find_close_points
    ii,io = find_close_points(ixyz, oxyz, overlap_distance)
    dup_wat_res = output_waters[io]	# Output water residues near input water residues
    new_wat_res = output_waters - dup_wat_res	# Output waters not near input waters
    dup_input_wat_res = input_waters[ii]	# Input waters near output waters

    return input_waters, new_wat_res, dup_wat_res, dup_input_wat_res

# also used in tool.py
def _water_residues(model):
    res = model.residues
    water_res = res[res.names == 'HOH']
    return water_res

