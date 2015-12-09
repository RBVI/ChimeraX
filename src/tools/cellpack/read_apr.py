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
                    from chimerax.core.geometry import Place
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
        if 'rep_file' in comp_details:
            comp_surfaces.append(comp_details['rep_file'])
        for interior_or_surface in ('interior', 'surface'):
            if interior_or_surface in comp_details:
                for ingr_name, ingr_info in comp_details[interior_or_surface]['ingredients'].items():
                    ingr_filename = ingr_info['include']
                    ingr_id = (comp_name, interior_or_surface, ingr_name)
                    ingr_filenames[ingr_id] = ingr_filename
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

def create_surfaces(session, surface_placements):

    from chimerax.core.surface.collada import read_collada_surfaces
    from os.path import basename
    from chimerax.core.geometry import Places
    surfs = []
    for path, placements in surface_placements:
        slist, msg = read_collada_surfaces(session, path, basename(path))
        surf = slist[0]
        if placements:
            # Surface geometry is in child drawings.
            p = Places(placements)
            for d in surf.child_drawings():
                d.positions = p * d.positions
            surf.name += ' %d copies' % len(placements)
#        print (surf.name, 'children', len(surf.child_drawings()), 'all children', len(surf.all_drawings()))
        surfs.append(surf)
    return surfs
