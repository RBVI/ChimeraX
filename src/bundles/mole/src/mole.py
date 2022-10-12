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
def read_mole_json(session, filename, name, transparency = 0):
    """Read Mole channels and create a separate Structure for each to be shown as spheres.

    :param filename: either the name of a file or a file-like object
    """

    if hasattr(filename, 'read'):
        # it's really a file-like object
        input = filename
    else:
        input = open(filename, 'r')

    import json
    j = json.load(input)

    models = channel_models(session, j['Channels'], transparency)

    if len(models) > 1:
        from chimerax.model_series.mseries import mseries_slider
        mseries_slider(session, models, title = f'{len(models)} channels', name = 'Channel')

    name  = f'{len(models)} channels'
    filename = channel_source(j['Config'])
    if filename:
        name += ' in ' + filename

    from chimerax.core.models import Model
    group = Model(name, session)
    group.add(models)

    message = f'Opened {len(models)} Mole channels'
    if filename:
        message += ' in ' + filename
    
    return [group], message

def channel_models(session, channel_json, transparency):
    channel_colors = mole_channel_colors()
    num_colors = len(channel_colors)

    opacity = int(255 * (1-transparency))
    
    models = []
    from chimerax.atomic import Structure
    for ctype in ('Pores', 'MergedPores', 'Tunnels', 'Paths'):
        channels = channel_json.get(ctype,[])
        for cnum, channel in enumerate(channels):
            chid = channel['Id']
            color = tuple(channel_colors[cnum % num_colors][:3]) + (opacity,)
            s = Structure(session, name = f'{ctype[:-1]} {chid}')
            s.display = (cnum == 0)
            res_name = 'T'
            chain_id = 'A'
            res_num = 1
            res = s.new_residue(res_name, chain_id, res_num)
            spheres = channel['Profile']
            for sphere in spheres:
                r,x,y,z = [sphere[attr] for attr in ('Radius', 'X', 'Y', 'Z')]
                a = s.new_atom(f's{chid}', 'C')
                a.coord = (x,y,z)
                a.radius = r
                a.draw_mode = a.SPHERE_STYLE
                a.color = color
                res.add_atom(a)
            models.append(s)

    return models

def channel_source(config_json):
    filename = ''
    file_path = dictionary_string(config_json, 'UserStructure')
    if file_path:
        filename = file_path.split('\\')[-1].split('/')[-1]
    else:
        pdb_id = (dictionary_string(config_json, 'PdbId') or
                  dictionary_string(config_json, 'Export', 'Parameters', 'ChimeraPDBId'))
        if pdb_id:
            filename = f'PDB {pdb_id}'
    return filename

def dictionary_string(d, *keys):
    for key in keys:
        if key in d:
            d = d[key]
        else:
            return None
    return d if isinstance(d, str) else None

def mole_channel_colors():
    channel_colors = [
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
    rgba8 = [BuiltinColors[cname].uint8x4() for cname in channel_colors]
    return rgba8

