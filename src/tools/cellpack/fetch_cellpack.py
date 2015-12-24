# -----------------------------------------------------------------------------
# Fetch AutoPack recipe files, ingredient files, and ingredient meshes.
#
default_autopack_database = 'https://github.com/mesoscope/cellPACK_data/raw/master/cellPACK_database_1.1.0'

# -----------------------------------------------------------------------------
#
def fetch_cellpack(session, cellpack_id, database = default_autopack_database, ignore_cache = False):

    try:
        path = fetch_autopack_results(session, cellpack_id, ignore_cache=ignore_cache)
        surf = fetch_autopack(session, path, cellpack_id, database, ignore_cache)
    except IOError as e:
        from chimerax.core.errors import UserError
        raise UserError('Unknown cellPACK id "%s"\n\n%s' % (results_name, str(e)))
    return [surf], 'Opened %s' % cellpack_id

# -----------------------------------------------------------------------------
#
def fetch_autopack(session, path, results_name, database = default_autopack_database,
                   ignore_cache = False):

    from . import read_apr
    recipe_loc, pieces = read_apr.read_autopack_results(path)
    recipe_url = recipe_loc.replace('autoPACKserver', database)
    from os.path import basename
    recipe_filename = basename(recipe_loc)
    from chimerax.core.fetch import fetch_file
    recipe_path = fetch_file(session, recipe_url, 'recipe for ' + results_name, recipe_filename, 'cellPACK',
                             ignore_cache=ignore_cache)

    ingr_filenames, comp_surfaces = read_apr.read_autopack_recipe(recipe_path)

    from chimerax.core.models import Model
    cpm = Model(results_name, session)

    # Fetch compartment surface files.
    csurfs = []
    from chimerax.core.surface.collada import read_collada_surfaces
    for comp_name, comp_loc, geom_loc in comp_surfaces:
        csurf = Model(comp_name, session)
        if comp_loc is not None:
            comp_url = comp_loc.replace('autoPACKserver', database)
            comp_filename = basename(comp_loc)
            comp_path = fetch_file(session, comp_url, 'compartment surface ' + comp_filename, comp_filename, 'cellPACK',
                                   ignore_cache=ignore_cache)
            slist, msg = read_collada_surfaces(session, comp_path, 'representation')
            csurf.add(slist)
        if geom_loc is not None:
            geom_url = geom_loc.replace('autoPACKserver', database)
            geom_filename = basename(geom_loc)
            geom_path = fetch_file(session, geom_url, 'compartment bounds ' + geom_filename, geom_filename, 'cellPACK',
                                   ignore_cache=ignore_cache)
            slist, msg = read_collada_surfaces(session, geom_path, 'geometry')
            for s in slist:
                s.display = False
            csurf.add(slist)
        csurfs.append(csurf)
    cpm.add(csurfs)

    # Added ingredient surfaces to compartments
    ingr_mesh_path = {}
    comp = {csurf.name:csurf for csurf in csurfs}
    ingr_ids = list(pieces.keys())
    ingr_ids.sort()	# Get reproducible ordering of ingredients
    for ingr_id in ingr_ids:
        ingr_filename = ingr_filenames[ingr_id]
        mesh_path = ingr_mesh_path.get(ingr_filename, None)
        if mesh_path is None:
            from urllib.parse import urljoin
            ingr_url = urljoin(recipe_url, ingr_filename)
            ingr_path = fetch_file(session, ingr_url, 'ingredient ' + ingr_filename, ingr_filename, 'cellPACK',
                                   ignore_cache=ignore_cache)
            mesh_loc = read_apr.read_ingredient(ingr_path)
            mesh_url = mesh_loc.replace('autoPACKserver', database)
            mesh_filename = basename(mesh_loc)
            mesh_path = fetch_file(session, mesh_url, 'mesh ' + mesh_filename, mesh_filename, 'cellPACK',
                                   ignore_cache=ignore_cache)
            ingr_mesh_path[ingr_filename] = mesh_path

        comp_name, interior_or_surf, ingr_name = ingr_id
        cs = comp.get((comp_name, interior_or_surf), None)
        if cs is None:
            cs = Model(interior_or_surf, session)
            comp[comp_name].add([cs])
            comp[(comp_name, interior_or_surf)] = cs
        placements = pieces[ingr_id]
        isurf = read_apr.create_surface(session, mesh_path, ingr_name, placements)
        cs.add([isurf])

    return cpm

# -----------------------------------------------------------------------------
# Fetch AutoPack results files
#
def fetch_autopack_results(session, results_name, database = default_autopack_database,
                           ignore_cache = False):

    # Fetch results file.
    results_url = database + '/results/%s.apr.json' % results_name
    session.logger.status('Fetching %s from web %s...' % (results_name,results_url))
    results_filename = results_name + '.apr.json'
    from chimerax.core.fetch import fetch_file
    results_path = fetch_file(session, results_url, 'results ' + results_name, results_filename, 'cellPACK',
                              ignore_cache=ignore_cache)
    return results_path

# -----------------------------------------------------------------------------
# Register to fetch cellPACK models.
#
def register_cellpack_fetch(session):
    from chimerax.core import fetch
    fetch.register_fetch(session, 'cellpack', fetch_cellpack, 'cellpack',
                         prefixes = ['cellpack'])

#  info_url = '%s/results/%%s.apr.json' % (default_autopack_database,)
# Example id 'HIV-1_0.1.6'
