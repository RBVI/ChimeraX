# -----------------------------------------------------------------------------
# Fetch AutoPack recipe files, ingredient files, and ingredient meshes.
#
default_autopack_database = 'https://github.com/mesoscope/cellPACK_data/raw/master/cellPACK_database_1.1.0'

# -----------------------------------------------------------------------------
#
def fetch_cellpack(session, cellpack_id):
    path = fetch_autopack_results(session, cellpack_id)
    surfs = fetch_results(session, path, cellpack_id)
    return surfs, 'Opened %s' % cellpack_id

# -----------------------------------------------------------------------------
#
def fetch_results(session, path, results_name, database = default_autopack_database, ignore_cache = False):
    try:
        surfs = fetch_autopack(session, path, results_name, database, ignore_cache)
    except IOError as e:
        from chimerax.core.errors import UserError
        raise UserError('Unknown cellPACK id "%s"\n\n%s' % (results_name, str(e)))
    return surfs

# -----------------------------------------------------------------------------
#
def fetch_autopack(session, path, results_name, database = default_autopack_database,
                   ignore_cache = False, check_certificates = False):

    from . import read_apr
    recipe_loc, pieces = read_apr.read_autopack_results(path)
    recipe_url = recipe_loc.replace('autoPACKserver', database)
    from os.path import basename
    recipe_filename = basename(recipe_loc)
    from chimerax.core.fetch import fetch_file
    recipe_path = fetch_file(session, recipe_url, 'recipe for ' + results_name, recipe_filename, 'cellPACK',
                             ignore_cache=ignore_cache, check_certificates=check_certificates)

    # Combine ingredients used in multiple compartments
    ingr_filenames, comp_surfaces = read_apr.read_autopack_recipe(recipe_path)
    ingr_placements = {}
    for ingr_id, placements in pieces.items():
        ingr_filename = ingr_filenames[ingr_id]
        if ingr_filename in ingr_placements:
            ingr_placements[ingr_filename].extend(placements)
        else:
            ingr_placements[ingr_filename] = list(placements)

    # Fetch ingredient surface files.
    surf_placements = []
    for ingr_filename, placements in ingr_placements.items():
        from urllib.parse import urljoin
        ingr_url = urljoin(recipe_url, ingr_filename)
        ingr_path = fetch_file(session, ingr_url, 'ingredient ' + ingr_filename, ingr_filename, 'cellPACK',
                               ignore_cache=ignore_cache, check_certificates=check_certificates)
        mesh_loc = read_apr.read_ingredient(ingr_path)
        mesh_url = mesh_loc.replace('autoPACKserver', database)
        mesh_filename = basename(mesh_loc)
        mesh_path = fetch_file(session, mesh_url, 'mesh ' + mesh_filename, mesh_filename, 'cellPACK',
                               ignore_cache=ignore_cache, check_certificates=check_certificates)
        surf_placements.append((mesh_path, placements))

    # Fetch compartment surface files.
    comp_paths = []
    for comp_loc in comp_surfaces:
        comp_url = comp_loc.replace('autoPACKserver', database)
        comp_filename = basename(comp_loc)
        comp_path = fetch_file(session, comp_url, 'component surface ' + comp_filename, comp_filename, 'cellPACK',
                               ignore_cache=ignore_cache, check_certificates=check_certificates)
        comp_paths.append(comp_path)

    # Open surface models
    surf_placements.extend((cmesh_path, []) for cmesh_path in comp_paths)
    surfs = read_apr.create_surfaces(session, surf_placements)

    return surfs

# -----------------------------------------------------------------------------
# Fetch AutoPack results files
#
def fetch_autopack_results(session, results_name, database = default_autopack_database,
                           ignore_cache = False, check_certificates = False):

    # Fetch results file.
    results_url = database + '/results/%s.apr.json' % results_name
    session.logger.status('Fetching %s from web %s...' % (results_name,results_url))
    results_filename = results_name + '.apr.json'
    from chimerax.core.fetch import fetch_file
    results_path = fetch_file(session, results_url, 'results ' + results_name, results_filename, 'cellPACK',
                              ignore_cache=ignore_cache, check_certificates=check_certificates)
    return results_path

# -----------------------------------------------------------------------------
# Register to fetch cellPACK models.
#
def register_cellpack_fetch(session):
    from chimera.core import fetch
    fetch.register_fetch(session, 'cellpack', fetch_cellpack, 'cellpack',
                         prefixes = ['cellpack'])

#  info_url = '%s/results/%%s.apr.json' % (default_autopack_database,)
# Example id 'HIV-1_0.1.6'
