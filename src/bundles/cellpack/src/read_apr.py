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

def read_autopack_results(path):

    j = read_json(path)
    recipe_path = j['recipe']['setupfile']
    pieces = {}
    for comp_name, cres in j['compartments'].items():
        for interior_or_surface, comp_ingr in cres.items():
            for ingr_name, ingr_places in comp_ingr['ingredients'].items():
                for translation, rotation44 in ingr_places['results']:
                    t0,t1,t2 = translation
                    r00,r01,r02 = rotation44[0][:3]
                    r10,r11,r12 = rotation44[1][:3]
                    r20,r21,r22 = rotation44[2][:3]
                    tf = ((r00,r01,r02,t0),
                          (r10,r11,r12,t1),
                          (r20,r21,r22,t2))
                    from chimerax.geometry import Place
                    p = Place(tf)
                    ingr_id = (comp_name, interior_or_surface, ingr_name)
                    if ingr_id in pieces:
                        pieces[ingr_id].append(p)
                    else:
                        pieces[ingr_id] = [p]

    return recipe_path, pieces

def read_autopack_recipe(path):

    j = read_json(path)
    ingr_filenames = {}
    comp_surfaces = []
    for comp_name, comp_details in j['compartments'].items():
        comp_surfaces.append((comp_name, comp_details.get('rep_file', None), comp_details.get('geom', None)))
        for interior_or_surface in ('interior', 'surface'):
            if interior_or_surface in comp_details:
                for ingr_name, ingr_info in comp_details[interior_or_surface]['ingredients'].items():
                    ingr_filename = ingr_info['include']
                    ingr_id = (comp_name, interior_or_surface, ingr_name)
                    ingr_filenames[ingr_id] = ingr_filename
    # TODO: Order compartment from outermost to innermost.  Order is not preserved in JSON file.
    comp_surfaces.sort()
    return ingr_filenames, comp_surfaces

def read_ingredient(path):

    j = read_json(path)
    return j['meshFile']
    
def read_json(path):

    f = open(path, 'r')
    import json
    j = json.load(f)
    f.close()
    return j

def create_surface(session, mesh_path, name, placements):

    from chimerax.surface.collada import read_collada_surfaces
    from chimerax.geometry import Places
    slist, msg = read_collada_surfaces(session, mesh_path, name)
    surf = slist[0]
    if placements:
        # Surface geometry is in child drawings.
        p = Places(placements)
        submodels = surf.child_drawings()
        if submodels:
            for d in submodels:
                d.positions = p * d.positions
                d.colors = copy_colors(d.colors, len(p))
        else:
            surf.positions = p
            surf.colors = copy_colors(surf.colors, len(p))
        surf.name += ' %d copies' % len(placements)
    return surf

def copy_colors(colors, n):
    if n == 1:
        return colors
    nc = len(colors)
    from numpy import empty
    c = empty((n,nc,4), colors.dtype)
    c[:] = colors
    return c.reshape((n*nc,4))
