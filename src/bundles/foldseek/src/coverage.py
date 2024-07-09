# vim: set expandtab ts=4 sw=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
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

def foldseek_coverage(session, conserved = 0):
    '''Show an image of all aligned sequences from a foldseek search, one sequence per image row.'''
    from .gui import foldseek_panel
    fp = foldseek_panel(session)
    if fp is None or len(fp.hits) == 0:
        from chimerax.core.errors import UserError
        raise UserError('No Foldseek results are shown')

    fcp = FoldseekCoveragePlot(session, fp.hits, fp.results_query_chain, conserved = conserved)
    return fcp

# -----------------------------------------------------------------------------
#
from chimerax.core.tools import ToolInstance
class FoldseekCoveragePlot(ToolInstance):

    name = 'Foldseek Sequence Coverage'
    help = 'help:user/tools/foldseek.html#coverage'

    def __init__(self, session, hits, query_chain, conserved = 0):

        self._hits = hits
        self._query_chain = query_chain
        self._conserved = conserved
        self._last_hover_xy = None

        ToolInstance.__init__(self, session, tool_name = self.name)

        from chimerax.ui import MainToolWindow
        tw = MainToolWindow(self)
        tw.fill_context_menu = self._fill_context_menu
        self.tool_window = tw
        parent = tw.ui_area

        from chimerax.ui.widgets import vertical_layout
        layout = vertical_layout(parent, margins = (5,0,0,0))

        from chimerax.ui.widgets import EntriesRow
        heading = f'Sequences for {len(hits)} Foldseek hits'
        hd = EntriesRow(parent, heading)
        self._heading = hd.labels[0]
        from Qt.QtWidgets import QSizePolicy
        hd.frame.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)  # Don't resize whole panel to fit heading
        layout.addWidget(hd.frame)

        rgb, self._sorted_hits, (qstart, qend) = coverage_image(hits, conserved = conserved)
        self._query_res = [(r.one_letter_code, r.number)
                           for r in query_chain.existing_residues[qstart-1:qend]]
        self._coverage_view = gv = CoverageView(parent, rgb, self._mouse_hover)
        layout.addWidget(gv)

        tw.manage(placement=None)	# Start floating
        
    # ---------------------------------------------------------------------------
    #
    def _mouse_hover(self, x, y):
        hit, res_type, res_num = self._hover_info(x, y)
        if hit and res_type:
            message = f'Hit {hit["database_full_id"]}   Query residue {res_type}{res_num}'
            self._last_hover_xy = x, y
        else:
            message = f'Sequence coverage for {len(self._hits)} Foldseek hits'
            self._last_hover_xy = None
        self._heading.setText(message)
        
    # ---------------------------------------------------------------------------
    #
    def _hover_info(self, x, y):
        if y >= 0 and y < len(self._sorted_hits) and x >= 0 and x < len(self._query_res):
            hit = self._sorted_hits[y]
            res_type, res_num = self._query_res[x]
        else:
            hit = res_type = res_num = None
        return hit, res_type, res_num

    # ---------------------------------------------------------------------------
    #
    def _fill_context_menu(self, menu, x, y):
        if self._last_hover_xy:
            # Use last hover position since menu post position is different by several pixels.
            hx, hy = self._last_hover_xy
            hit, res_type, res_num = self._hover_info(hx, hy)
        if hit:
            menu.addAction(f'Open structure {hit["database_full_id"]}',
                           lambda hit=hit: self._open_hit(hit))
        if res_type:
            menu.addAction(f'Select query residue {res_type}{res_num}',
                           lambda res_num=res_num: self._select_query_residue(res_num))

        menu.addAction('Save image', self._save_image)
        
    # ---------------------------------------------------------------------------
    #
    def _open_hit(self, hit):
        from .gui import foldseek_panel
        fp = foldseek_panel(self.session)
        kw = {'trim': fp.trim, 'alignment_cutoff_distance': fp.alignment_cutoff_distance} if fp else {}
        from .foldseek import open_hit
        open_hit(self.session, hit, self._query_chain, **kw)

    # ---------------------------------------------------------------------------
    #
    def _select_query_residue(self, res_num):
        resspec = self._query_chain.string(style = 'command') + f':{res_num}'
        from chimerax.core.commands import run
        run(self.session, f'select {resspec}')
    
    # ---------------------------------------------------------------------------
    #
    def _save_image(self, default_suffix = '.png'):
        from os import path, getcwd
        suggested_path = path.join(getcwd(), 'coverage' + default_suffix)
        from Qt.QtWidgets import QFileDialog
        parent = self.tool_window.ui_area
        save_path, ftype  = QFileDialog.getSaveFileName(parent, 'Foldseek Coverage Image', suggested_path)
        if save_path:
            if not path.splitext(save_path)[1]:
                save_path += default_suffix
            self._coverage_view.save_image(save_path)

# ---------------------------------------------------------------------------
#
from Qt.QtWidgets import QGraphicsView
class CoverageView(QGraphicsView):
    def __init__(self, parent, rgb, hover_callback = None):
        self._hover_callback = hover_callback

        QGraphicsView.__init__(self, parent)
        from Qt.QtWidgets import QSizePolicy
        from Qt.QtCore import Qt
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        from Qt.QtWidgets import QGraphicsScene
        self._scene = gs = QGraphicsScene(self)
        self.setScene(gs)

        self._pixmap_item = None
        self.set_image(rgb)

        # Report residues and atoms as mouse hovers over plot.
        if hover_callback:
            self.setMouseTracking(True)

    def resizeEvent(self, event):
        # Rescale histogram when window resizes
        self.fitInView(self.sceneRect())
        QGraphicsView.resizeEvent(self, event)

    def mouseMoveEvent(self, event):
        if self._hover_callback:
            p = self.mapToScene(event.pos())
            x,y = p.x(), p.y()
            self._hover_callback(int(x),int(y))

    def set_image(self, rgb):
        scene = self.scene()
        pi = self._pixmap_item
        if pi is not None:
            scene.removeItem(pi)

        self._pixmap = pixmap = pixmap_from_rgb(rgb)
        self._pixmap_item = scene.addPixmap(pixmap)
        scene.setSceneRect(0, 0, pixmap.width(), pixmap.height())

    def save_image(self, path):
        self._pixmap.save(path)

# -----------------------------------------------------------------------------
#
def coverage_image(hits, conserved = 0, conserved_color = (255,0,0), identity_color = (0,255,0)):
    qstarts = []
    qends = []
    for hit in hits:
        qstarts.append(hit['qstart'])
        qends.append(hit['qend'])
    qstart, qend = min(qstarts), max(qends)
    qlen = qend-qstart+1

    shits = sorted_hits(hits)
    from numpy import zeros, uint8, array
    cover = zeros((len(hits), qlen), uint8)
    for i,hit in enumerate(shits):
        query_coverage(hit, qstart, cover[i,:])
    
    colors = array(((255,255,255), (0,0,0), identity_color), uint8)
    rgb = colors[cover]

    if conserved:
        for i in range(qlen):
            ci = cover[:,i]
            ns,nd = (ci == 2).sum(), (ci == 1).sum()
            if ns > conserved * (ns + nd):
                rgb[ci==2,i,:] = conserved_color
        
    return rgb, shits, (qstart, qend)

# -----------------------------------------------------------------------------
#
def sorted_hits(hits):
    from numpy import array, float32
    intervals = array([(hit['qstart'], hit['qend']) for hit in hits], float32)

    # Cluster intervals using kmeans
    from scipy.cluster.vq import kmeans, vq
    for k in range(1,20):
        codebook, distortion = kmeans(intervals, k)
        if distortion <= 20:
            break

    # Order clusters longest interval first
    centers = list(codebook)
    centers.sort(key = lambda se: se[0]-se[1])
    labels, dist = vq(intervals, centers)

    # Sort by cluster and within a cluster by start of interval
    i = list(range(len(hits)))
    i.sort(key = lambda j: (labels[j], hits[j]['qstart']))
    shits = [hits[j] for j in i]

    return shits
    
# -----------------------------------------------------------------------------
#
def query_coverage(hit, cover_start, cover):
    qaln, taln = hit['qaln'], hit['taln']
    qi = hit['qstart']
    for qaa, taa in zip(qaln, taln):
        if qaa != '-' and taa != '-':
            cover[qi-cover_start] = 2 if taa == qaa else 1
        if qaa != '-':
            qi += 1

# -----------------------------------------------------------------------------
#
def pixmap_from_rgb(rgb):
    # Save image to a PNG file
    from Qt.QtGui import QImage, QPixmap
    h, w = rgb.shape[:2]
    im = QImage(rgb.data, w, h, 3*w, QImage.Format_RGB888)
    pixmap = QPixmap.fromImage(im)
    return pixmap
    
def register_foldseek_coverage_command(logger):
    from chimerax.core.commands import CmdDesc, register, FloatArg
    from chimerax.atomic import ChainArg
    desc = CmdDesc(
        required = [],
        keyword = [('conserved', FloatArg)],
        synopsis = 'Show an image of all aligned sequences from a foldseek search, one sequence per image row.'
    )
    register('foldseek coverage', desc, foldseek_coverage, logger=logger)
