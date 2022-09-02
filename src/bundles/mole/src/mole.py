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
mole: Read Mole tunnel files
============================

Read Mole tunnel json files and display as spheres.
"""

# -----------------------------------------------------------------------------
#
def read_mole_json(session, filename, name, transparency = 0.5):
    """Read Mole tunnels and create a separate Structure for each to be shown as spheres.

    :param filename: either the name of a file or a file-like object
    """

    if hasattr(filename, 'read'):
        # it's really a file-like object
        input = filename
    else:
        input = open(filename, 'r')

    import json
    j = json.load(input)
    tunnels = j['Channels']['Tunnels']

    tunnel_colors = mole_tunnel_colors()
    num_colors = len(tunnel_colors)

    opacity = int(255 * (1-transparency))
    
    models = []
    from chimerax.atomic import Structure
    for tnum, tunnel in enumerate(tunnels):
        tid = tunnel['Id']
        color = tuple(tunnel_colors[tnum % num_colors][:3]) + (opacity,)
        s = Structure(session, name = f'Tunnel {tid}')
        s.display = (tnum == 0)
        res_name = 'T'
        chain_id = 'A'
        res_num = 1
        res = s.new_residue(res_name, chain_id, res_num)
        spheres = tunnel['Profile']
        for sphere in spheres:
            r,x,y,z = [sphere[attr] for attr in ('Radius', 'X', 'Y', 'Z')]
            a = s.new_atom(f's{tid}', 'C')
            a.coord = (x,y,z)
            a.radius = r
            a.draw_mode = a.SPHERE_STYLE
            a.color = color
            res.add_atom(a)
        models.append(s)

    from chimerax.model_series.mseries import mseries_slider
    mseries_slider(session, models, title = f'{len(models)} tunnels', name = 'Tunnel')
    
    return models, f'Opened {len(models)} Mole tunnels' 

def mole_tunnel_colors():
    tunnel_colors = [
        'orange red',
        'orange',
        'yellow',
        'green',
        'forest green',
        'cyan',
        'light sea green',
        'blue',
        'cornflower blue',
        'medium blue',
        'purple',
        'hot pink',
        'magenta',
        'spring green',
        'plum',
        'sky blue',
        'goldenrod',
        'olive drab',
        'coral',
        'rosy brown',
        'slate gray',
        'red',
        ]
    from chimerax.core.colors import BuiltinColors
    rgba8 = [BuiltinColors[cname].uint8x4() for cname in tunnel_colors]
    return rgba8

