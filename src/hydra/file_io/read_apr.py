def open_autopack_results(path, session):
    '''
    Open an Autopack results files (.apr suffix) and create surfaces
    for each component of the model.
    '''
    pieces = read_apr_file(path)
    surfs = create_surfaces(path, pieces, session)
    return surfs

def read_apr_file(path):

    import sys
    f = open(path, 'r')
    lines = f.readlines()
    f.close()

    pieces = {}
    from ..geometry.place import Place
    for line in lines:
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

    return pieces

def print_pieces(pieces):

    flist = pieces.keys()
    flist.sort()
    for fn in flist:
        print (fn, len(pieces[fn]))

def create_surfaces(apr_path, pieces, session):

    not_found = set()
    pdbs = []
    from os.path import dirname, basename, join, exists
    dir = dirname(apr_path)
    fnames = list(pieces.keys())
    fnames.sort()
    surfs = []
    for fname in fnames:
        path_prefix = join(dir, fname)
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
        from . import collada
        surf = collada.read_collada_surfaces(path, session)
        for p in surf.surface_pieces():
            if p.copies:
                p.copies = sum([[pl1*pl2 for pl1 in tflist] for pl2 in p.copies], [])
            else:
                p.copies = tflist
    return surf

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
