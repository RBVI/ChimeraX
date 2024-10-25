# vim: set expandtab ts=4 sw=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This particular file is part of the ChimeraX library. You can also
# redistribute and/or modify it under the terms of the GNU Lesser General
# Public License version 2.1 as published by the Free Software Foundation.
# For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
# EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
# LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
# VERSION 2.1
#
# This notice must be embedded in or attached to all copies, including partial
# copies, of the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

def install_umap(session):
    try:
        import umap
    except ModuleNotFoundError:
        session.logger.info('Installing umap-learn package from PyPi')
        from chimerax.core.commands import run
        run(session, 'pip install umap-learn')

def umap_embed(data, random_seed = 0):
    if data.shape[0] <= 2:
        from chimerax.core.errors import UserError
        raise UserError(f'UMAP requires at least 3 data points, got {data.shape[0]}')
    n_neighbors = min(15, data.shape[0]-1) # Avoid warning when fewer data points then default n_neighbors value
    init = 'spectral'
    if data.shape[0] <= data.shape[1] + 1:
        init = 'random'  # Default spectral initialization fails if number of components > number of samples.
    import umap
    reducer = umap.UMAP(n_neighbors = n_neighbors, init = init, random_state = random_seed, n_jobs = 1)
    mapper = reducer.fit(data)
    return reducer.embedding_

def k_means_clusters(data, k):
    from scipy.cluster.vq import kmeans, vq
    codebook, distortion = kmeans(data, k)
    labels, dist = vq(data, codebook)
    return labels

def cluster_by_distance(umap_xy, cluster_distance):
    if len(umap_xy) <= 1:
        from numpy import ones, int32
        return ones((len(umap_xy),), int32)
    from scipy.cluster.hierarchy import linkage, fcluster
    Z = linkage(umap_xy)
    cluster_numbers = fcluster(Z, cluster_distance, criterion = 'distance')
    return cluster_numbers

def color_by_cluster(cluster_numbers):
    from chimerax.core.colors import random_colors
    ccolors = random_colors(max(cluster_numbers)+1)
    colors = ccolors[cluster_numbers]
    return colors

from chimerax.interfaces.graph import Graph
class UmapPlot(Graph):
    def __init__(self, session, title = 'UMAP Plot', tool_name = 'Umap'):
        self._have_colors = False
        self._node_area = 500	# Area in pixels
        nodes = edges = []
        Graph.__init__(self, session, nodes, edges, title = title, tool_name = tool_name)
        self.font_size = 5	# Override graph default value of 12 points

    def set_nodes(self, names, umap_xy, colors = None):
        self._have_colors = (colors is not None)
        self.nodes = self._make_nodes(names, umap_xy, colors)
        self.graph = self._make_graph()
        self.draw_graph()

    def _make_nodes(self, names, umap_xy, colors = None):
        from chimerax.interfaces.graph import Node
        nodes = []
        for i, (name, xy) in enumerate(zip(names, umap_xy)):
            n = Node()
            n.name = name
            n.position = (xy[0], xy[1], 0)
            n.size = self._node_area
            if colors is not None:
                n.color = tuple(r/255 for r in colors[i])
            nodes.append(n)
        return nodes

    def layout_projection(self):
        from chimerax.geometry import identity
        return identity()
