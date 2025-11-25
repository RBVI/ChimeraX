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

def similar_structures_sequences(session, show_conserved = True, conserved_threshold = 0.5,
                                 conserved_color = (225,190,106), identity_color = (64,176,166),
                                 lddt_coloring = False, order = 'cluster or evalue', from_set = None):
    '''Show an image of all aligned sequences from a similar structure search, one sequence per image row.'''
    from .simstruct import similar_structure_results
    results = similar_structure_results(session, from_set)
    if results is None:
        from chimerax.core.errors import UserError
        raise UserError('No similar structure results are open')

    sp = _existing_sequence_plot(session, results)
    if sp:
        sp.display(True)
        return sp
    
    conserved_color = conserved_color[:3]	# Don't use transparency
    identity_color = identity_color[:3]
    sp = SequencePlotPanel(session, results, order = order,
                           show_conserved = show_conserved, conserved_threshold = conserved_threshold,
                           conserved_color = conserved_color, identity_color = identity_color,
                           lddt_coloring = lddt_coloring)
    return sp

# -----------------------------------------------------------------------------
#
def _existing_sequence_plot(session, results):
    for tool in session.tools.list():
        if isinstance(tool, SequencePlotPanel) and tool._results is results:
            return tool
    return None

# -----------------------------------------------------------------------------
#
from chimerax.core.tools import ToolInstance
class SequencePlotPanel(ToolInstance):

    name = 'Sequence Plot'
    help = 'help:user/tools/foldseek.html#sequences'

    def __init__(self, session, results, order = 'cluster or evalue',
                 show_conserved = True, conserved_threshold = 0.5,
                 conserved_color = (225,190,106), identity_color = (64,176,166),
                 lddt_coloring = False):

        self._results = results
        self._query_chain = results.query_chain
        self._order = order
        self._hit_order_array = None
        self._show_conserved = show_conserved
        self._conserved_threshold = conserved_threshold
        self._conserved_color = conserved_color
        self._identity_color = identity_color
        self._color_by_lddt = lddt_coloring
        self._last_hover_xy = None

        ToolInstance.__init__(self, session, tool_name = self.name)

        from chimerax.ui import MainToolWindow
        tw = MainToolWindow(self)
        tw.fill_context_menu = self._fill_context_menu
        self.tool_window = tw
        parent = tw.ui_area
        parent.mousePressEvent = parent.contextMenuEvent		# Show context menu on left click also.

        from chimerax.ui.widgets import vertical_layout
        layout = vertical_layout(parent, margins = (5,0,0,0))

        from chimerax.ui.widgets import EntriesRow
        heading = f'Sequences for {results.num_hits} Foldseek hits'
        hd = EntriesRow(parent, heading)
        self._heading = hd.labels[0]
        from Qt.QtWidgets import QSizePolicy
        hd.frame.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)  # Don't resize whole panel to fit heading
        layout.addWidget(hd.frame)

        rgb = self._sequence_image()
        self._sequence_plot = sp = SequencePlot(parent, rgb, self._mouse_hover)
        layout.addWidget(sp)

        results.set_coverage_attribute()
        results.set_conservation_attribute()
        results.set_entropy_attribute()

        tw.manage(placement=None)	# Start floating

    # ---------------------------------------------------------------------------
    #
    def _update_sequence_image(self):
        rgb = self._sequence_image()
        self._sequence_plot.set_image(rgb)
        
    # ---------------------------------------------------------------------------
    #
    def _sequence_image(self):
        alignment_array = self._results.sequence_alignment_array()
        hit_order = self._hit_order()
        rgb = _sequence_image(alignment_array, hit_order, identity_color = self._identity_color)
        if self._color_by_lddt:
            self._color_lddt(rgb)
        if self._show_conserved:
            _color_conserved(alignment_array, hit_order, rgb,
                             threshold = self._conserved_threshold,
                             conserved_color = self._conserved_color,
                             identity_color = self._identity_color)
        return rgb

    # ---------------------------------------------------------------------------
    #
    def _hit_order(self):
        if self._hit_order_array is None:
            order = self._order
            hits = self._results.hits
            if order == 'cluster':
                hit_order, num_clusters = _hits_cluster_order(hits)
            elif order == 'evalue':
                hit_order = _hits_order_by_attribute(hits, 'evalue')
            elif order == 'cluster or evalue':
                hit_order, num_clusters = _hits_cluster_order(hits)
                if num_clusters == 1:
                    hit_order = _hits_order_by_attribute(hits, 'evalue')
            elif order in 'identity':
                hit_order = _hits_order_by_attribute(hits, 'pident', smallest_first = False)
            elif order in 'lddt':
                lddt = self._results.lddt_scores()
                from numpy import argsort, count_nonzero
                mean_lddt = lddt.sum(axis=1) / count_nonzero(lddt, axis=1)
                hit_order = argsort(mean_lddt)[::-1]
            else:
                from numpy import arange
                hit_order = arange(len(hits), dtype=int32)
            self._hit_order_array = hit_order
        return self._hit_order_array

    # ---------------------------------------------------------------------------
    #
    def _mouse_hover(self, x, y):
        hit, res_type, res_num = self._hover_info(x, y)
        if hit:
            message = f'Hit {hit["database_full_id"]}'
            if res_type:
                message += f'     Query residue {res_type}{res_num}'
            self._last_hover_xy = x, y
        else:
            nhits = self._results.num_hits
            message = f'Sequence plot for {nhits} Foldseek hits'
            self._last_hover_xy = None
        self._heading.setText(message)
        
    # ---------------------------------------------------------------------------
    #
    def _hover_info(self, x, y):
        r = self._results
        if r.query_chain is None:
            return None, None, None
        query_res = r.query_residue_names()
        if y >= 0 and y < r.num_hits and x >= 0 and x < len(query_res):
            order = self._hit_order()
            hit = r.hits[order[y]]
            rname = query_res[x]
            res_type, res_num = (None, None) if rname is None else rname
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
                db_id = hit["database_full_id"]
                menu.addAction(f'Open structure {db_id}', lambda hit=hit: self._open_hit(hit))
                menu.addAction(f'Show {db_id} in table', lambda hit=hit: self._show_hit_in_table(hit))
            if res_type:
                menu.addAction(f'Select query residue {res_type}{res_num}',
                               lambda res_num=res_num: self._select_query_residue(res_num))
            menu.addSeparator()

        menu.addAction('Order sequences by:', lambda: 0)
        menu.addAction('   e-value', lambda: self._order_by('evalue'))
        menu.addAction('   cluster', lambda: self._order_by('cluster'))
        menu.addAction('   identity', lambda: self._order_by('identity'))
        menu.addAction('   mean LDDT', lambda: self._order_by('lddt'))

        menu.addSeparator()
        self._add_menu_toggle(menu, 'Color conserved', self._show_conserved, self._color_conserved)
        self._add_menu_toggle(menu, 'Color by LDDT', self._color_by_lddt, self._show_lddt)

        menu.addSeparator()
        menu.addAction('Color query structure by:', lambda: 0)	# Section header
        menu.addAction('    coverage', self._color_query_coverage)
        menu.addAction('    conservation', self._color_query_conservation)
        menu.addAction('    highly conserved', self._color_query_highly_conserved)
        menu.addAction('    local alignment', self._color_query_lddt)
                
        menu.addSeparator()
        menu.addAction('Save image', self._save_image)
        
    # ---------------------------------------------------------------------------
    #
    def _add_menu_toggle(self, menu, text, checked, callback):
        from Qt.QtGui import QAction
        a = QAction(text, menu)
        a.setCheckable(True)
        a.setChecked(checked)
        a.triggered.connect(callback)
        menu.addAction(a)
        
    # ---------------------------------------------------------------------------
    #
    def _open_hit(self, hit):
        self._results.open_hit(self.session, hit)
        
    # ---------------------------------------------------------------------------
    #
    def _show_hit_in_table(self, hit):
        from .gui import similar_structures_panel
        ssp = similar_structures_panel(self.session)
        if ssp and self._results is ssp.results:
            ssp.select_table_row(hit)

    # ---------------------------------------------------------------------------
    #
    def _select_query_residue(self, res_num):
        resspec = self._query_chain.string(style = 'command') + f':{res_num}'
        from chimerax.core.commands import run
        run(self.session, f'select {resspec}')

    # ---------------------------------------------------------------------------
    #
    def _order_by(self, ordering_name):
        self._order = ordering_name
        self._hit_order_array = None
        self._update_sequence_image()

    # ---------------------------------------------------------------------------
    #
    def _color_conserved(self, show = True):
        self._show_conserved = show
        self._update_sequence_image()

    # ---------------------------------------------------------------------------
    #
    def _show_lddt(self, show = True):
        if show:
            r = self._results
            if not r.have_c_alpha_coordinates():
                from .coords import similar_structures_fetch_coordinates
                if not similar_structures_fetch_coordinates(self.session, ask = True, from_set = r.name):
                    return

        self._color_by_lddt = show
        if show:
            self._show_conserved = False
        self._update_sequence_image()
        
    # ---------------------------------------------------------------------------
    #
    def _color_lddt(self, rgb):
        cmap = self._lddt_colormap()
        lddt_scores = self._results.lddt_scores()
        order = self._hit_order()
        nhits = len(lddt_scores)
        for i,h in enumerate(order):
            smask = (lddt_scores[h] > 0)
            rgba = cmap.interpolated_rgba8(lddt_scores[h,smask])
            rgb[i,smask,:] = rgba[:,:3]

    # ---------------------------------------------------------------------------
    #
    def _lddt_colormap(self):
        from chimerax.core.colors import BuiltinColors, Colormap
        color_names = ('red', 'orange', 'yellow', 'cornflowerblue', 'blue')
        colors = [BuiltinColors[name] for name in color_names]
        scores = (0, 0.2, 0.4, 0.6, 0.8)
        cmap = Colormap(scores, colors)
        return cmap

    # ---------------------------------------------------------------------------
    #
    def _color_query_coverage(self):
        if self._query_chain:
            qspec = self._query_chain.string(style = 'command')
            alignment_array = self._results.sequence_alignment_array()
            from numpy import count_nonzero
            n = count_nonzero(alignment_array[1:], axis=0).max()  # Max coverage
            self._run_command(f'color byattribute r:coverage {qspec} palette 0,red:{n//2},white:{n},blue')
    def _color_query_conservation(self):
        if self._query_chain:
            qspec = self._query_chain.string(style = 'command')
            self._run_command(f'color byattribute r:conservation {qspec} palette 0,blue:0.25,white:0.5,red')
    def _color_query_highly_conserved(self):
        if self._query_chain:
            qspec = self._query_chain.string(style = 'command')
            threshold = self._conserved_threshold
            self._run_command(f'color {qspec} gray')
            self._run_command(f'color {qspec} & ::conservation>={threshold} red')
    def _color_query_lddt(self):
        if self._query_chain:
            self._results.set_lddt_attribute()
            qspec = self._query_chain.string(style = 'command')
            self._run_command(f'color byattribute r:lddt {qspec} palette 0,red:0.2,orange:0.4,yellow:0.6,cornflowerblue:0.8,blue')
    def _run_command(self, command):
        from chimerax.core.commands import run
        run(self.session, command)

    # ---------------------------------------------------------------------------
    #
    def _save_image(self, default_suffix = '.png'):
        from os import path, getcwd
        suggested_path = path.join(getcwd(), 'sequences' + default_suffix)
        from Qt.QtWidgets import QFileDialog
        parent = self.tool_window.ui_area
        save_path, ftype  = QFileDialog.getSaveFileName(parent, 'Sequence Plot', suggested_path)
        if save_path:
            if not path.splitext(save_path)[1]:
                save_path += default_suffix
            self._sequence_plot.save_image(save_path)

# ---------------------------------------------------------------------------
#
from Qt.QtWidgets import QGraphicsView
class SequencePlot(QGraphicsView):
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
def _sequence_image(alignment_array, order, no_align_color = (255,255,255),
                    align_color = (0,0,0), identity_color = (254,254,98)):
    hits_array = alignment_array[1:,:]	# First row is query sequence
    # Make a 2D array with values 0=unaligned, 1=aligned, 2=identical.
    # This avoids 2D masks that don't work well in numpy.
    from numpy import array, uint8
    aligned_mask = (hits_array != 0)
    identical_mask = (hits_array == alignment_array[0])
    res_type = aligned_mask.astype(uint8) + identical_mask.astype(uint8)
    colors = array((no_align_color, align_color, identity_color), uint8)
    rgb = colors[res_type[order,:]]
    return rgb

# -----------------------------------------------------------------------------
#
def _color_conserved(alignment_array, order, rgb, threshold = 0.3, min_seqs = 10,
                     conserved_color = (225,190,106), identity_color = (64,176,166)):
    seq_len = alignment_array.shape[1]
    from numpy import count_nonzero
    for i in range(seq_len):
        if alignment_array[0,i] == 0:
            continue
        mi = (alignment_array[1:,i] == alignment_array[0,i])
        ni = mi.sum()	# Number of sequences with matching amino acid at column i
        na = count_nonzero(alignment_array[1:,i])  # Number of sequences aligned at column i
        color = conserved_color if na >= min_seqs and ni >= threshold * na else identity_color
        rgb[mi[order],i,:] = color

# -----------------------------------------------------------------------------
#
def _hits_cluster_order(hits):
    from numpy import array, float32, int32
    intervals = array([(hit['qstart'], hit['qend']) for hit in hits], float32)

    # Cluster intervals using kmeans
    from scipy.cluster.vq import kmeans, vq
    for k in range(1,20):
        codebook, distortion = kmeans(intervals, k)
        if distortion <= 20:
            break
    num_clusters = k

    # Order clusters longest interval first
    centers = list(codebook)
    centers.sort(key = lambda se: se[0]-se[1])
    labels, dist = vq(intervals, centers)

    # Sort by cluster and within a cluster by start of interval
    i = list(range(len(hits)))
    i.sort(key = lambda j: (labels[j], hits[j]['qstart']))
    order = array(i, int32)

    return order, num_clusters

# -----------------------------------------------------------------------------
#
def _hits_order_by_attribute(hits, attribute, smallest_first = True):
    from numpy import array, float32, argsort
    evalues = array([hit[attribute] for hit in hits], float32)
    order = argsort(evalues)
    return order if smallest_first else order[::-1]

# -----------------------------------------------------------------------------
#
def pixmap_from_rgb(rgb):
    # Save image to a PNG file
    from Qt.QtGui import QImage, QPixmap
    h, w = rgb.shape[:2]
    im = QImage(rgb.data, w, h, 3*w, QImage.Format_RGB888)
    pixmap = QPixmap.fromImage(im)
    return pixmap
    
def register_similar_structures_sequences_command(logger):
    from chimerax.core.commands import CmdDesc, register, FloatArg, BoolArg, EnumOf, Color8Arg, StringArg
    from chimerax.atomic import ChainArg
    desc = CmdDesc(
        required = [],
        keyword = [('show_conserved', BoolArg),
                   ('conserved_threshold', FloatArg),
                   ('conserved_color', Color8Arg),
                   ('identity_color', Color8Arg),
                   ('lddt_coloring', BoolArg),
                   ('order', EnumOf(['cluster', 'evalue', 'identity', 'lddt'])),
                   ('from_set', StringArg)],
        synopsis = 'Show an image of all aligned sequences from a similar structure search, one sequence per image row.'
    )
    register('similarstructures sequences', desc, similar_structures_sequences, logger=logger)
