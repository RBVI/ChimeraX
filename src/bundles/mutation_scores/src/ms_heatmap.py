# vim: set expandtab shiftwidth=4 softtabstop=4:

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

# -----------------------------------------------------------------------------
#
from chimerax.core.tools import ToolInstance
class MutationScoresHeatmap(ToolInstance):

    help = 'https://www.rbvi.ucsf.edu/chimerax/data/mutation-scores-oct2024/mutation_scores.html'

    def __init__(self, session, tool_name = 'Mutation Scores Heatmap'):
        
        ToolInstance.__init__(self, session, tool_name)

        from chimerax.ui import MainToolWindow
        tw = MainToolWindow(self)
        tw.fill_context_menu = self._fill_context_menu
        self.tool_window = tw
        parent = tw.ui_area

        from chimerax.ui.widgets import vertical_layout
        layout = vertical_layout(parent, margins = (5,0,0,0))

        self._score_view = gv = ScoreView(parent, self._report_cell_info)
        from Qt.QtWidgets import QSizePolicy
        from Qt.QtCore import Qt
#        gv.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
#        gv.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
#        gv.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        layout.addWidget(gv, stretch=1)

        from Qt.QtWidgets import QGraphicsScene
        self._scene = gs = QGraphicsScene(gv)
        gs.setSceneRect(0, 0, 500, 500)
        gv.setScene(gs)

        from Qt.QtWidgets import QLabel
        self._info_label = info = QLabel(parent)
        layout.addWidget(info)

        self._set_heatmap_image()

        tw.manage(placement=None)	# Start floating

    # ---------------------------------------------------------------------------
    #
    def closed(self):
        return self.tool_window.tool_instance is None

    # ---------------------------------------------------------------------------
    #
    def _fill_context_menu(self, menu, x, y):
        menu.addAction('Select residue', self._select_residue)
        menu.addAction('Save image', self._save_image)
    
    # ---------------------------------------------------------------------------
    #
    def _set_heatmap_image(self):
        score_matrix = self._score_matrix()
        blue,white,red = (0,0,1,1), (1,1,1,1), (1,0,0,1)
        from chimerax.core.colors import Colormap
        colormap = Colormap((-2.0,-1.0,1.0,2.0), (blue, white, white, red))
        self._score_view._make_image(score_matrix, colormap)

    # ---------------------------------------------------------------------------
    #
    _amino_acids = 'HRKDEFWYNQILCSTVMAGP'
    def _score_matrix(self):
        from .ms_data import mutation_all_scores
        msets = mutation_all_scores(self.session)
        self._mutation_set = mset = msets[0]  # TODO: Allow choosing mutation set
        scores = None
        score_names = mset.score_names()
        # TODO: Allow choosing score names
        score_names = [score_name for score_name in score_names if score_name.endswith('_effect')]
        self._score_names = score_names
        self._num_scores = score_count = len(score_names)
        aa_to_index = {aa:i for i, aa in enumerate(self._amino_acids)}
        self._res_aa = res_aa = {}
        for snum, score_name in enumerate(score_names):
            score_values = mset.score_values(score_name)
            if scores is None:
                # TODO: This may not give maximum res number
                self._num_residues = rmax = max(score_values.residue_numbers())
                from numpy import zeros, float32
                self._scores = scores = zeros((rmax, 20, score_count), float32)
            sscores = scores[:,:,snum]
            for res_num, from_aa, to_aa, value in score_values.all_values():
                res_aa[res_num] = from_aa
                aa_index = aa_to_index[to_aa]
                sscores[res_num-1, aa_index] = value
            mean, sdev = score_values.synonymous_mean_and_sdev()
            sscores -= mean
            sscores /= sdev

        scores_2d = scores.reshape((rmax, 20*score_count)).transpose()
        return scores_2d

    # ---------------------------------------------------------------------------
    #
    def _report_cell_info(self, column_index, row_index):
        num_cols = self._num_residues
        num_rows = 20 * self._num_scores
        if column_index < 0 or row_index < 0 or column_index >= num_cols or row_index >= num_rows or column_index+1 not in self._res_aa:
            msg = ''
        else:
            res_num = column_index + 1
            from_aa = self._res_aa[res_num]
            score_num = row_index % self._num_scores
            score_name = self._score_names[score_num]
            aa_index = row_index // self._num_scores
            to_aa = self._amino_acids[aa_index]
            score_value = self._scores[res_num-1, aa_index, score_num]
            msg = f'{from_aa}{res_num}{to_aa} {score_name} {"%.2f"%score_value}'
        self._info_label.setText(msg)

    # ---------------------------------------------------------------------------
    #
    def _save_image(self, default_suffix = '_heatmap.png'):
        from os.path import dirname, join
        filename = self._mutation_set.name + default_suffix
        dir = dirname(self._mutation_set.path)
        suggested_path = join(dir, filename)
        from Qt.QtWidgets import QFileDialog
        parent = self.tool_window.ui_area
        path, ftype  = QFileDialog.getSaveFileName(parent,
                                                   'Mutation Heatmap Image',
                                                   suggested_path)
        if path:
            self._score_view.save_image(path)
        
    # ---------------------------------------------------------------------------
    #
    def _show_help(self):
        from chimerax.core.commands import run
        run(self.session, 'help %s' % self.help)

# ---------------------------------------------------------------------------
#
from Qt.QtWidgets import QGraphicsView
class ScoreView(QGraphicsView):
    def __init__(self, parent, report_cell_info_cb=None):
        QGraphicsView.__init__(self, parent)
        self._report_cell_info_callback = report_cell_info_cb
        self._pixmap_item = None

        # Report cell info as mouse hovers over plot.
        self.setMouseTracking(True)

        # Zoom in
        self.scale(2,2)

    def sizeHint(self):
        from Qt.QtCore import QSize
        return QSize(500,500)

    def mouseMoveEvent(self, event):
        if self._report_cell_info_callback:
            x,y = self._scene_position(event)
            self._report_cell_info_callback(int(x),int(y))

    def _scene_position(self, event):
        p = self.mapToScene(event.pos())
        return p.x(), p.y()

    def _make_image(self, matrix, colormap):
        scene = self.scene()
        pi = self._pixmap_item
        if pi is not None:
            scene.removeItem(pi)

        rgb = matrix_to_rgb(matrix, colormap)
        pixmap = rgb_to_pixmap(rgb)
        self._pixmap_item = scene.addPixmap(pixmap)
        scene.setSceneRect(0, 0, pixmap.width(), pixmap.height())

    def save_image(self, path):
        pixmap = self.grab()
        pixmap.save(path)

# -----------------------------------------------------------------------------
#
def matrix_to_rgb(matrix, colormap):
    rgb_flat = colormap.interpolated_rgba8(matrix.ravel())[:,:3]
    n,m = matrix.shape
    rgb = rgb_flat.reshape((n,m,3)).copy()
    return rgb

# -----------------------------------------------------------------------------
#
def rgb_to_pixmap(rgb):
    # Save image to a PNG file
    from Qt.QtGui import QImage, QPixmap
    h, w = rgb.shape[:2]
    im = QImage(rgb.data, w, h, 3*w, QImage.Format_RGB888)
    pixmap = QPixmap.fromImage(im)
    return pixmap

def mutation_heatmap(session):
    hm = MutationScoresHeatmap(session)
    return hm

def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, StringArg, BoolArg
    desc = CmdDesc(
        synopsis = 'Show a heatmap of mutation scores.'
    )
    register('mutationscores heatmap', desc, mutation_heatmap, logger=logger)
