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

# -----------------------------------------------------------------------------
#
def read_pae_matrix(path):
    if path.endswith('.json'):
        return read_json_pae_matrix(path)
    elif path.endswith('.pkl'):
        return read_pickle_pae_matrix(path)
    else:
        from chimerax.core.errors import UserError
        raise UserError(f'AlphaFold predicted aligned error (PAE) files must be in JSON (*.json) or pickle (*.pkl) format, {path} unrecognized format')

# -----------------------------------------------------------------------------
#
def read_json_pae_matrix(path):
    '''Open AlphaFold database distance error PAE JSON file returning a numpy matrix.'''
    f = open(path, 'r')
    import json
    j = json.load(f)
    f.close()
    d = j[0]

    # Read distance errors into numpy array
    from numpy import array, zeros, float32, int32
    r1 = array(d['residue1'], dtype=int32)
    r2 = array(d['residue2'], dtype=int32)
    ea = array(d['distance'], dtype=float32)
    me = d['max_predicted_aligned_error']
    n = r1.max()
    pae = zeros((n,n), float32)
    pae[r1-1,r2-1] = ea

    return pae

# -----------------------------------------------------------------------------
#
def read_pickle_pae_matrix(path):
    f = open(path, 'rb')
    import pickle
    p = pickle.load(f)
    f.close()
    if isinstance(p, dict) and 'predicted_aligned_error' in p:
        return p['predicted_aligned_error']

    from chimerax.core.errors import UserError
    raise UserError(f'File {path} does not contain AlphaFold predicted aligned error (PAE) data')

# -----------------------------------------------------------------------------
#
def pae_rgb(pae_matrix):

    # Create an rgba image showing values.
    me = pae_matrix.max()
    dmin,dmax = 0,me
    ec = pae_matrix - dmin
    ec /= (dmax-dmin)
    from numpy import empty, uint8, clip, sqrt
    clip(ec, 0, 1, ec)

    # Match AlphaFold DB heatmap colors
    n = pae_matrix.shape[0]
    rgb = empty((n,n,3), uint8)
    rgb[:,:,0] = 30 + 225*(ec*ec)
    rgb[:,:,1] = 70 + 185*sqrt(ec)
    rgb[:,:,2] = rgb[:,:,0]

    return rgb

# -----------------------------------------------------------------------------
#
def pae_pixmap(rgb):
    # Save image to a PNG file
    from Qt.QtGui import QImage, QPixmap
    h, w = rgb.shape[:2]
    im = QImage(rgb.data, w, h, 3*w, QImage.Format_RGB888)
    pixmap = QPixmap.fromImage(im)
    return pixmap

# -----------------------------------------------------------------------------
#
def pae_image(rgb):
    from PIL import Image
    pi = Image.fromarray(rgb)
    #pi.save('test.png')      # Save image to a PNG file
    return pi

# -----------------------------------------------------------------------------
# Code take from Tristan Croll, ChimeraX ticket #4966.
#
# https://github.com/tristanic/isolde/blob/master/isolde/src/reference_model/alphafold/find_domains.py
#
def pae_domains(pae_matrix, pae_power=1, pae_cutoff=5, graph_resolution=0.5):
    # PAE matrix is not strictly symmetric.
    # Prediction for error in residue i when aligned on residue j may be different from 
    # error in j when aligned on i. Take the smallest error estimate for each pair.
    import numpy
    pae_matrix = numpy.minimum(pae_matrix, pae_matrix.T)
    weights = 1/pae_matrix**pae_power if pae_power != 1 else 1/pae_matrix

    import networkx as nx
    g = nx.Graph()
    size = weights.shape[0]
    g.add_nodes_from(range(size))
    edges = numpy.argwhere(pae_matrix < pae_cutoff)
    # Limit to bottom triangle of matrix
    edges = edges[edges[:,0]<edges[:,1]]
    sel_weights = weights[edges.T[0], edges.T[1]]
    wedges = [(i,j,w) for (i,j),w in zip(edges,sel_weights)]
    g.add_weighted_edges_from(wedges)

    from networkx.algorithms.community import greedy_modularity_communities
    clusters = greedy_modularity_communities(g, weight='weight', resolution=graph_resolution)
    return clusters

# -----------------------------------------------------------------------------
#
def color_by_pae_domain(residues, clusters, colors = None, min_cluster_size=10):
    if colors is None:
        from chimerax.core.colors import random_colors
        colors = random_colors(len(clusters), seed=0)

    from numpy import array, int32
    for c, color in zip(clusters, colors):
        if len(c) >= min_cluster_size:
            cresidues = residues[array(list(c),int32)]
            cresidues.ribbon_colors = color
            cresidues.atoms.colors = color

# -----------------------------------------------------------------------------
#
def alphafold_pae(session, path, model = None):
    '''Load AlphaFold predicted aligned error file and show heatmap.'''
    if model is None:
        from chimerax.atomic import all_atomic_structures
        models = all_atomic_structures(session)
        if len(models) != 1:
            from chimerax.core.errors import UserError
            raise UserError('Must specify which AlphaFold structure to associate with PAE data using "model" option.')
        model = models[0]
    from .heatmap_gui import AlphaFoldHeatmap
    hm = AlphaFoldHeatmap(session, 'AlphaFold Heatmap')
    hm.set_heatmap(path, model)

# -----------------------------------------------------------------------------
#
def register_alphafold_pae_command(logger):
    from chimerax.core.commands import CmdDesc, register, OpenFileNameArg
    from chimerax.atomic import AtomicStructureArg
    desc = CmdDesc(
        required = [('path', OpenFileNameArg)],
        keyword = [('model', AtomicStructureArg)],
        synopsis = 'Show AlphaFold predicted aligned error as heatmap'
    )
    
    register('alphafold pae', desc, alphafold_pae, logger=logger)
