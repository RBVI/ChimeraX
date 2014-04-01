def open_autopack_results(path, session):
    '''
    Open an Autopack results files (.apr suffix) and create surfaces
    for each component of the model.
    '''
    surfs = []
    recpath, pieces = parse_apr_file(path)
    if recpath:
        from os.path import dirname, join
        rp = join(dirname(path), '..', 'recipes', recpath)
        rsurfs = read_recipe_file(rp, session)
        surfs.extend(rsurfs)

    surfs.extend(create_surfaces(path, pieces, session))

    use_short_names(surfs)

    return surfs

def parse_apr_file(path):

    import sys
    f = open(path, 'r')
    lines = f.readlines()
    f.close()

    recpath = None
    pieces = {}
    from ..geometry.place import Place
    for line in lines:
        if line.startswith('#'):
            rprefix = '# recipe '
            if line.startswith(rprefix):
                recpath = line[len(rprefix):].strip()
            continue
        fields = line.replace('<','').replace('>','').split(',')
        if len(fields) < 23:
            continue
        t0,t1,t2 = tuple(float(x) for x in fields[0:3])
        m = tuple(float(x) for x in fields[3:19])
        r00,r01,r02 = m[0:3]
        r10,r11,r12 = m[4:7]
        r20,r21,r22 = m[8:11]
        tf = Place(((r00,r01,r02,t0),
                    (r10,r11,r12,t1),
                    (r20,r21,r22,t2)))
        unk1 = fields[19]
        fname = fields[20]
        unk2 = fields[21]
        unk3 = fields[22]
        if fname in pieces:
            pieces[fname].append(tf)
        else:
            pieces[fname] = [tf]

    return recpath, pieces

def print_pieces(pieces):

    flist = pieces.keys()
    flist.sort()
    for fn in flist:
        print (fn, len(pieces[fn]))

def create_surfaces(apr_path, pieces, session):

    not_found = set()
    pdbs = []
    from os.path import dirname, basename, join, exists
    gdir = join(dirname(apr_path), '..', 'geometries')
    fnames = list(pieces.keys())
    fnames.sort()
    surfs = []
    for fname in fnames:
        path_prefix = join(gdir, fname)
        tflist = pieces[fname]
        if len(tflist) == 0:
            continue
#        pdb_path = path_prefix + '.pdb'
#        if exists(pdb_path):
#            pdbs.append((pdb_path, tflist))
#        else:
        descrip = '%s, %d copies' % (fname, len(tflist))
        surf = create_surface_copies(path_prefix, tflist, session)
        if surf:
            surfs.append(surf)
        else:
            not_found.add(fname)
            descrip += ', file not found'

        print(descrip)

#    make_multiscale_models(pdbs)

    return surfs

def create_surface_copies(path_prefix, tflist, session):

    path = path_prefix + '.stl'
    from os.path import exists
    if exists(path):
        from .read_stl import read_stl
        surf = read_stl(path, session)
        p = surf.surface_pieces()[0]
        p.color = color = random_color(surf.name)
    else:
        path = path_prefix + '.dae'
        if not exists(path):
            return None
        surf = read_collada_surface(path, session)

    for p in surf.surface_pieces():
        if p.copies:
            p.copies = sum([[pl1*pl2 for pl1 in tflist] for pl2 in p.copies], [])
        else:
            p.copies = tflist

    return surf

def read_collada_surface(path, session):

    from . import collada
    surf = collada.read_collada_surfaces(path, session)
    if hasattr(surf, 'collada_unit_name') and surf.collada_unit_name in ('meter', None):
        # TODO: If unit meter tag omitted in file PyCollada sets unit name to None.
        #  Probably should patch pycollada to return unit name even if unit meter scale factor not given.
        scale_vertices(surf.plist, 100)
    if is_cinema4d_collada_surface(surf):
        fix_cinema4d_coordinates(surf)       # Correct for Cinema4d coordinates
    return surf

def scale_vertices(splist, scale):
    for p in splist:
        va, ta = p.geometry
        va *= scale
        p.geometry = va, ta

def is_cinema4d_collada_surface(surf):
    if not hasattr(surf, 'collada_contributors'):
        return False
    for c in surf.collada_contributors:
        a = c.authoring_tool
        if a and a.startswith('CINEMA4D'):
            return True
    return False

def fix_cinema4d_coordinates(surf):
    for p in surf.surface_pieces():
        v, t = p.geometry
        n = p.normals

        vc,nc = v.copy(), n.copy()
        vc[:,0] = -v[:,2]
        vc[:,2] = v[:,0]
        nc[:,0] = -n[:,2]
        nc[:,2] = n[:,0]
        p.geometry = vc, t
        p.normals = nc

def make_multiscale_models(pdbs):

    if len(pdbs) == 0:
        return

    from chimera import openModels
    from matrix import chimera_xform
    import MultiScale
    mm = MultiScale.multiscale_manager()
    for pdb_path, tflist in pdbs:
        pdb = openModels.open(pdb_path)[0]
        # Multiscale keeps the original pdb unmoved.
        # Make it align with surfaces.
        pdb.openState.localXform(chimera_xform(tflist[0]))
        mgroup = mm.molecule_multimer(pdb, tflist)
        multiscale_single_color([mgroup], random_color(pdb.name))

    MultiScale.show_multiscale_model_dialog()

def multiscale_single_color(mpieces, color):

    from MultiScale import find_pieces, Chain_Piece
    for cp in find_pieces(mpieces, Chain_Piece):
        cp.set_color(color)

def random_color(seed = None):

    import random
    if not seed is None:
        if isinstance(seed, str):
            # Make 64-bit machines and 32-bit produce the same 32-bit seed
            # by casting 64-bit hash values to signed 32-bit.
            seed = string_hash(seed)
        random.seed(seed)
    from random import uniform
    return (uniform(.2,1), uniform(.2,1), uniform(.2,1), 1)

# Hash that produces the same value across sessions and platforms.
# Python 3 hash() returns different values for the same string in different sessions.
def string_hash(s):
  h = 0
  hmax = 2**32
  for c in s:
    h = (ord(c) + (h << 6) + (h << 16) - h) % hmax
  return h

def read_ingredient_file(xml_path, session):
    from .opensave import open_from_database
    from .fetch import fetch_from_database
    from xml.dom.minidom import parse
    from os.path import basename, dirname, join
    t = parse(xml_path)
    models = []
    for i in t.getElementsByTagName('ingredient'):
        mf = i.getAttribute('meshFile')
        if mf and mf.endswith('.dae'):
            mfr = join(dirname(xml_path),'..','geometries',basename(mf))
            models.append(read_collada_surface(mfr, session))
        sf = i.getAttribute('sphereFile')
        if sf:
            sfr = join(dirname(xml_path),'..','collisionTrees',basename(sf))
            models.append(read_sphere_file(sfr, session))
        pdb_id = i.getAttribute('pdb')
        if pdb_id and len(pdb_id) == 4:
            models.extend(fetch_from_database(pdb_id, 'PDB', session))
    return models

def read_sphere_file(path, session):
    f = open(path)
    lines = f.readlines()
    f.close()
    
    for s,line in enumerate(lines):
        if line.startswith('# x y z r of spheres'):
            break

    xyzr = []
    for line in lines[s:]:
        fields = line.split()
        if len(fields) == 4:
            try:
                xyzr.append(tuple(float(x) for x in fields))
            except ValueError:
                pass

    n = len(xyzr)
    from numpy import array, float32, ones, uint8, empty, arange
    xyzra = array(xyzr, float32)
    xyz = xyzra[:,:3].copy()
    r = xyzra[:,3].copy()
    element_nums = ones((n,), uint8)
    chain_ids = empty((n,), 'S1')
    chain_ids[:] = 'A'
    res_nums = arange(1,n+1)
    res_names = empty((n,), 'S3')
    res_names[:] = "S"
    atom_names = empty((n,), 'S3')
    atom_names[:] = 's'
    from ..molecule import Molecule
    m = Molecule(path, xyz, element_nums, chain_ids, res_nums, res_names, atom_names)
    m.radii = r
    m.color = (180,180,180,128)
    m.color_mode = 'custom'
    return m

def read_recipe_file(recipe_path, session):
    from xml.dom.minidom import parse
    from os.path import basename, dirname, join
    t = parse(recipe_path)
    models = []
    for c in t.getElementsByTagName('compartment'):
        rf = c.getAttribute('rep_file')
        if rf and rf.endswith('.dae'):
            rfr = join(dirname(recipe_path),'..','geometries',basename(rf))
            models.append(read_collada_surface(rfr, session))
    return models

# Strip common prefix and numeric suffixes from surface names.
def use_short_names(surfs):
    from os.path import commonprefix, splitext
    plen = len(commonprefix([s.name for s in surfs]))
    for s in surfs:
        s.name = splitext(s.name[plen:])[0].rstrip('_0123456789')
