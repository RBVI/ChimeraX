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

        self._score_pad = 1	# Number of blank pixels after each score group
        self._color_residue_on_hover = False
        self._residue_highlight_color = (255,255,0,255)
        self._last_highlight_residues = None		# (Residues, colors)
        
        ToolInstance.__init__(self, session, tool_name)

        from chimerax.ui import MainToolWindow
        tw = MainToolWindow(self)
        tw.fill_context_menu = self._fill_context_menu
        self.tool_window = tw
        parent = tw.ui_area

        from chimerax.ui.widgets import vertical_layout
        layout = vertical_layout(parent, margins = (5,0,0,0))

        # Make heatmap widget
        self._score_view = sv = self._create_graphics_pane(parent)
        layout.addWidget(sv, stretch=1)

        # Buttons, e.g. Options, Help
        bf = self._create_action_buttons(parent)
        layout.addWidget(bf)

        # Status line that shows mutation mouse hovers over.
        from Qt.QtWidgets import QLabel
        self._info_label = info = QLabel(parent)
        bf.layout().insertWidget(2, self._info_label)

        # Options panel
        options = self._create_options_gui(parent)
        layout.addWidget(options)

        # Draw the heatmap and axis labels
        self._draw_graphics()

        tw.manage(placement=None)	# Start floating
        
    # ---------------------------------------------------------------------------
    #
    def _create_graphics_pane(self, parent):
        gv = ScoreView(parent, self._report_cell_info)
        return gv
    
    # ---------------------------------------------------------------------------
    #
    def _create_action_buttons(self, parent):
        from chimerax.ui.widgets import button_row
        f = button_row(parent,
                       [('Options', self._show_or_hide_options),
                        ('Help', self._show_help)],
                       spacing = 10)
        return f

    # ---------------------------------------------------------------------------
    #
    def _draw_graphics(self):
        scene = self._score_view.scene
        scene.clear()
        self._set_heatmap_image()
        self._make_residue_axis_labels()
        self._make_amino_acid_axis_labels()
        scene.setSceneRect(scene.itemsBoundingRect())

    # ---------------------------------------------------------------------------
    #
    def closed(self):
        return self.tool_window.tool_instance is None

    # ---------------------------------------------------------------------------
    #
    def _fill_context_menu(self, menu, x, y):

        if self._have_structure:
            from Qt.QtCore import QPointF
            gxy = self.tool_window.ui_area.mapToGlobal(QPointF(x,y))
            gv_point = self._score_view.mapFromGlobal(gxy)
            ix, iy = self._score_view.graphics_view_to_image_position(gv_point)
            res_num, from_aa, to_aa, score_name, score_value = self._cell_info(ix, iy)
            if res_num is not None and from_aa is not None:
                menu.addAction(f'Select residue {from_aa}{res_num}',
                               lambda res_num=res_num: self._select_residue(res_num))

            _add_menu_toggle(menu, 'Color structure residue while hovering',
                             self._color_residue_on_hover, self._color_on_hover_changed)

        menu.addAction('Save image', self._save_image)

    # ---------------------------------------------------------------------------
    #
    @property
    def _have_structure(self):
        return len(self._mutation_set.associate_chains(self.session)) > 0
    
    # ---------------------------------------------------------------------------
    #
    def _set_heatmap_image(self):
        score_matrix = self._score_matrix()
        colormap = self._colormap()
        rgb = matrix_to_rgb(score_matrix, colormap)
        self._heatmap_height = rgb.shape[0]

        # Add divider lines
        divider_line_color = (0,0,0) if self._num_scores > 5 else (255,255,255)
        row_step = self._num_scores + self._score_pad
        for i in range(self._num_scores,rgb.shape[0],row_step):
            rgb[i:i+self._score_pad,:,:] = divider_line_color

        if self._highlight_structure_residues.enabled:
            mset = self._mutation_set
            if len(mset.associate_chains(self.session)) > 0:
                res, rnums = mset.associated_residues(tuple(range(1, self._num_residues+1)))
                struct_res_nums = set(rnums)
                no_struct_res_nums = [r for r in range(1, self._num_residues+1) if not r in struct_res_nums]
                for r in no_struct_res_nums:
                    rgb[:,r-1,:] = _shade_gray(rgb[:,r-1,:])

        pixels_per_cell = self._pixels_per_cell.value
        self._score_view.set_image(rgb, pixels_per_cell)

    # ---------------------------------------------------------------------------
    #
    @property
    def _score_names(self, exclude = ['position']):
        mset = self._mutation_set
        score_names = [score_name for score_name in mset.score_names() if score_name not in exclude]
        sf = self._score_name_filter.value
        snames = sf.strip().split(',')
        if snames:
            matches = []
            for sname in snames:
                if sname.startswith('*'):
                    suffix = sname[1:]
                    matches.extend([score_name for score_name in score_names if score_name.endswith(suffix)])
                elif sname in score_names:
                    matches.append(sname)
            if matches:
                score_names = matches
        return score_names

    # ---------------------------------------------------------------------------
    #
    @property
    def _all_score_names(self):
        mset = self._mutation_set
        return mset.score_names() if mset else []
        
    # ---------------------------------------------------------------------------
    #
    _amino_acids = 'HRKDEFWYNQILCSTVMAGP'
    def _score_matrix(self):
        scores = None
        mset = self._mutation_set
        self._num_scores = score_count = len(self._score_names)
        aa_to_index = {aa:i for i, aa in enumerate(self._amino_acids)}
        self._res_aa = res_aa = {}
        subtract_fit = self._subtract_fit
        sub_score_values = mset.score_values(subtract_fit) if subtract_fit else None
        for snum, score_name in enumerate(self._score_names):
            score_values = mset.score_values(score_name)
            if sub_score_values:
                score_values = score_values.subtract_fit(sub_score_values)
            if scores is None:
                # TODO: This may not give maximum res number
                res_nums = score_values.residue_numbers()
                self._rmin, self._rmax = rmin, rmax = min(res_nums), max(res_nums)
                self._num_residues = num_res = rmax-rmin+1
                from numpy import zeros, float32
                self._scores = scores = zeros((num_res, 20, score_count + self._score_pad), float32)
            sscores = scores[:,:,snum]
            for res_num, from_aa, to_aa, value in score_values.all_values():
                res_aa[res_num] = from_aa
                aa_index = aa_to_index[to_aa]
                sscores[res_num-rmin, aa_index] = value
            if self._normalize.enabled:
                mean, sdev = score_values.synonymous_mean_and_sdev()
                sscores -= mean
                sscores /= sdev

        scores_2d = scores.reshape((num_res, 20*(score_count+self._score_pad))).transpose()[:-1,:]
        return scores_2d

    # ---------------------------------------------------------------------------
    #
    def _report_cell_info(self, column_index, row_index):
        res_num, from_aa, to_aa, score_name, score_value = self._cell_info(column_index, row_index)
        if score_name is None:
            msg = ''
        else:
            msg = f'{from_aa}{res_num}{to_aa} {score_name} {"%.2f"%score_value}'
        self._info_label.setText(msg)

        if res_num is not None and self._color_residue_on_hover and self._have_structure:
            self._uncolor_highlight_residues()
            res, rnums = self._mutation_set.associated_residues([res_num])
            if len(res) > 0:
                self._last_highlight_residues = (res, res.ribbon_colors)
                res.ribbon_colors = self._residue_highlight_color
            
    # ---------------------------------------------------------------------------
    #
    def _cell_info(self, column_index, row_index):
        # Float column index ranges from 0-1 from left to right edge of first pixel.
        c, r = int(round(column_index-0.5)), int(round(row_index-0.5))
        res_num = from_aa = to_aa = score_name = score_value = None
        num_cols = self._num_residues
        num_rows = 20 * (self._num_scores + self._score_pad)
        rmin = self._rmin
        if c >= 0 and c < num_cols and r >= 0 and r < num_rows and c+rmin in self._res_aa:
            res_num = c + rmin
            from_aa = self._res_aa[res_num]
            score_num = r % (self._num_scores + self._score_pad)
            if score_num < self._num_scores:
                score_name = self._score_names[score_num]
                aa_index = r // (self._num_scores + self._score_pad)
                to_aa = self._amino_acids[aa_index]
                score_value = self._scores[res_num-rmin, aa_index, score_num]
        return res_num, from_aa, to_aa, score_name, score_value

    # ---------------------------------------------------------------------------
    #
    def _select_residue(self, res_num):
        mset = self._mutation_set
        mset.associate_chains(self.session)
        res,rnums = mset.associated_residues([res_num])
        if len(res) == 0:
            self.session.logger.info(f'No associated structure residues for residue number {res_num}')
        else:
            from chimerax.core.commands import run
            from chimerax.atomic import concise_residue_spec
            spec = concise_residue_spec(self.session, res)
            run(self.session, f'select {spec}')

    # ---------------------------------------------------------------------------
    #
    def _color_on_hover_changed(self, enable):
        self._color_residue_on_hover = enable
        if not enable:
            self._uncolor_highlight_residues()

    # ---------------------------------------------------------------------------
    #
    def _uncolor_highlight_residues(self):
        if self._last_highlight_residues:
            last_res, last_colors = self._last_highlight_residues
            last_res.ribbon_colors = last_colors
            self._last_highlight_residues = None

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
    def _make_residue_axis_labels(self, residue_step = None):
        pixels_per_cell = self._pixels_per_cell.value
        if residue_step is None:
            if pixels_per_cell == 1:
                residue_step = 100
            elif pixels_per_cell == 2:
                residue_step = 50
            elif pixels_per_cell in (3,4):
                residue_step = 20
            else:
                residue_step = 10
        rmin, rmax = self._rmin, self._rmax
        imin, imax = (rmin//residue_step) + 1, ((rmax-1)//residue_step)
        rnums = [rmin] + [i*residue_step for i in range(imin, imax+1)] + [rmax]
        for r in rnums: 
            text = str(r)
            t = self._score_view.scene.addText(text)
            rect = t.boundingRect()
            x = (r-rmin+.5)*pixels_per_cell - rect.width()/2
            y = self._heatmap_height * pixels_per_cell
            t.setPos(x, y)

    # ---------------------------------------------------------------------------
    #
    def _make_amino_acid_axis_labels(self):
        pixels_per_cell = self._pixels_per_cell.value
        num_scores = self._num_scores
        scores_height = num_scores * pixels_per_cell
        aa_step = (num_scores + self._score_pad) * pixels_per_cell
        for i, aa in enumerate(self._amino_acids):
            t = self._score_view.scene.addText(aa)
            rect = t.boundingRect()
            x = -rect.width()
            y = i*aa_step + 0.5*scores_height - rect.height()/2
            t.setPos(x, y)
        
    # ---------------------------------------------------------------------------
    #
    def _create_options_gui(self, parent):
        from chimerax.ui.widgets import CollapsiblePanel
        self._options_panel = p = CollapsiblePanel(parent, title = None)
        f = p.content_area

        # Which mutation set
        from chimerax.ui.widgets import EntriesRow, ColorButton
        ms = EntriesRow(f, 'Mutations', ('set1', 'set2'))
        self._mutation_set_menu = msm = ms.values[0]
        from .ms_data import mutation_all_scores
        msets = mutation_all_scores(self.session)
        if msets:
            msm.value = msets[0].name
        menu = msm.widget.menu()
        menu.aboutToShow.connect(self._mutation_set_menu_about_to_show)
        menu.triggered.connect(self._draw_graphics)

        # Which scores.  Comma-separated list.  *_effect includes all scores with suffix
        sc = EntriesRow(f, 'Score names', '')
        self._score_name_filter = scf = sc.values[0]
        scf.pixel_width = 300
        scf.return_pressed.connect(self._draw_graphics)

        # Zoom factor for heatmap
        zf = EntriesRow(f, 'Pixels per cell', 1)
        self._pixels_per_cell = ppc = zf.values[0]
        ppc.return_pressed.connect(self._draw_graphics)
        
        # Colormap
        self._colormaps = {}
        cm = EntriesRow(f, 'Colormap', -2.0, ColorButton, -1.0, ColorButton, 1.0, ColorButton, 2.0,
                        ('Default', self._set_default_colormap))
        v1,c1,v2,c2,v3,c3,v4 = cm.values
        c1.color, c2.color, c3.color = 'blue', 'white', 'red'
        self._colormap_values = (v1,v2,v3,v4)
        self._colormap_colors = (c1,c2,c2,c3)
        for cv in (v1,v2,v3,v4):
            cv.format = '%.2g'
            cv.return_pressed.connect(self._changed_colormap)
        for cc in (c1,c2,c3):
            cc.color_changed.connect(self._changed_colormap)

        # Normalize scores
        ns = EntriesRow(f, True, 'Normalize scores to synonymous mean 0, standard deviation 1')
        self._normalize = nv = ns.values[0]
        nv.changed.connect(self._normalize_changed)

        # Subtract fit
        sf = EntriesRow(f, False, 'Subtract fit of score', ('', 'score2'))
        self._use_subtract_fit, self._subtract_score = sfe, sfs = sf.values
        score_names = self._all_score_names
        if score_names:
            surf_names = [score_name for score_name in score_names if 'surface' in score_name.lower()]
            sfs.value = surf_names[0] if surf_names else score_names[0]
        sfe.changed.connect(self._subtract_fit_changed)
        menu = sfs.widget.menu()
        menu.aboutToShow.connect(self._subtract_fit_menu_about_to_show)
        menu.triggered.connect(self._subtract_fit_changed)

        # Highlight mutations with associated structure residues
        hs = EntriesRow(f, True, 'Highlight structure residues')
        self._highlight_structure_residues = hs = hs.values[0]
        hs.changed.connect(self._set_heatmap_image)

        return p
        
    # ---------------------------------------------------------------------------
    #
    @property
    def _mutation_set(self):
        mset_name = self._mutation_set_menu.value
        from .ms_data import mutation_scores
        mset = mutation_scores(self.session, mset_name)
        return mset

    # ---------------------------------------------------------------------------
    #
    def _mutation_set_menu_about_to_show(self):
        menu = self._mutation_set_menu.widget.menu()
        menu.clear()
        from .ms_data import mutation_scores_names
        for ms_name in mutation_scores_names(self.session):
            menu.addAction(ms_name)

    # ---------------------------------------------------------------------------
    #
    def _changed_colormap(self):
        self._set_heatmap_image()
    
    # ---------------------------------------------------------------------------
    #
    def _set_default_colormap(self):
        self._switch_colormap(use_default = True)
        self._set_heatmap_image()
        
    # ---------------------------------------------------------------------------
    #
    def _colormap(self):
        values = [cv.value for cv in self._colormap_values]
        from chimerax.core.colors import rgba8_to_rgba, Colormap
        colors = [rgba8_to_rgba(cc.color) for cc in self._colormap_colors]
        colormap = Colormap(values, colors)
        self._colormaps[self._colormap_name] = [values, colors]
        return colormap
    
    # ---------------------------------------------------------------------------
    #
    @property
    def _colormap_name(self):
        normalized = self._normalize.enabled
        if normalized:
            cmap_name = 'normalized'
        elif self._subtract_fit is None:
            cmap_name = 'unnormalized'
        else:
            cmap_name = 'unnormalized subtract fit'
        return cmap_name
    
    # ---------------------------------------------------------------------------
    #
    def _switch_colormap(self, use_default = False):
        cmap_name = self._colormap_name
        if use_default or cmap_name not in self._colormaps:
            self._colormaps[cmap_name] = self._default_colormap()
        values, colors = self._colormaps[cmap_name]
        for cv, v in zip(self._colormap_values, values):
            cv.value = v
        for cc, c in zip(self._colormap_colors, colors):
            cc.color = c

    # ---------------------------------------------------------------------------
    #
    def _default_colormap(self):
        if self._normalize.enabled:
            cmap = [(-2,-1,1,2), ('blue', 'white', 'white', 'red')]
        else:
            mean, sd = self._all_score_mean_and_sdev()
            cmap = [(mean-2*sd, mean-sd, mean+sd, mean+2*sd),
                    ('blue', 'white', 'white', 'red')]
        return cmap

    # ---------------------------------------------------------------------------
    #
    def _all_score_mean_and_sdev(self):
        values = []
        mset = self._mutation_set
        subtract_fit = self._subtract_fit
        sub_score_values = mset.score_values(subtract_fit) if subtract_fit else None
        for score_name in self._score_names:
            score_values = mset.score_values(score_name)
            if sub_score_values:
                score_values = score_values.subtract_fit(sub_score_values)
            for res_num, from_aa, to_aa, value in score_values.all_values():
                values.append(value)
        from numpy import mean, std
        return mean(values), std(values)

    # ---------------------------------------------------------------------------
    #
    def _normalize_changed(self):
        self._switch_colormap()
        self._set_heatmap_image()

    # ---------------------------------------------------------------------------
    #
    def _subtract_fit_changed(self):
        self._switch_colormap()
        self._set_heatmap_image()

    # ---------------------------------------------------------------------------
    #
    @property
    def _subtract_fit(self):
        if not self._use_subtract_fit.enabled:
            return None
        score_name = self._subtract_score.value
        return score_name

    # ---------------------------------------------------------------------------
    #
    def _subtract_fit_menu_about_to_show(self):
        mset = self._mutation_set
        menu = self._subtract_score.widget.menu()
        for score_name in mset.score_names():
            menu.addAction(score_name)

    # ---------------------------------------------------------------------------
    #
    def _show_or_hide_options(self):
        self._options_panel.toggle_panel_display()

    # ---------------------------------------------------------------------------
    #
    def _show_help(self):
        from chimerax.core.commands import run
        run(self.session, 'help %s' % self.help)
        
# ---------------------------------------------------------------------------
#
def _add_menu_toggle(menu, text, checked, callback):
    from Qt.QtGui import QAction
    a = QAction(text, menu)
    a.setCheckable(True)
    a.setChecked(checked)
    a.triggered.connect(callback)
    menu.addAction(a)

# ---------------------------------------------------------------------------
#
def _shade_gray(rgb_array, gray = (100,100,100)):
    from numpy import clip, int32
    return clip(rgb_array.astype(int32) - gray, 0, 255).astype(rgb_array.dtype)
    
# ---------------------------------------------------------------------------
#
from Qt.QtWidgets import QGraphicsView
class ScoreView(QGraphicsView):
    def __init__(self, parent, report_cell_info_cb=None):
        QGraphicsView.__init__(self, parent)
        self._report_cell_info_callback = report_cell_info_cb
        self._pixmap_item = None

        from Qt.QtWidgets import QGraphicsScene
        self.scene = gs = QGraphicsScene(self)
        self.setScene(gs)

        # Report cell info as mouse hovers over plot.
        self.setMouseTracking(True)

        # Zoom in
#        self.scale(2,2)

    def sizeHint(self):
        rect = self.scene.itemsBoundingRect()
        size = rect.size().toSize()
        return size

    def mouseMoveEvent(self, event):
        if self._report_cell_info_callback is None:
            return
        # This gives event handler is for QAbstractScrollArea and the event is in viewport() coordinates.
        # We need the event in QGraphicsView coordinates which differ by 1 pixel in x,y.
        gv_point = self.mapFromGlobal(event.globalPosition())  # Map from viewport to graphics view.
        x,y = self.graphics_view_to_image_position(gv_point)
        self._report_cell_info_callback(x,y)

    def graphics_view_to_image_position(self, gv_point):
        # QGrahicsView.mapToScene() takes integer QPoint, not float QPointF.
        #   p = self.mapToScene(gv_point)
        # So use the viewportTransform() to use floating point coordinates.
        v2s, invertible = self.viewportTransform().inverted()
        p = v2s.map(gv_point)	# Map float view position to scene
        ip = self._pixmap_item.mapFromScene(p)
        return ip.x(), ip.y()

    def set_image(self, rgb, pixels_per_cell = 1):
        scene = self.scene
        pi = self._pixmap_item
        if pi is not None and pi in self.scene.items():
            scene.removeItem(pi)

        pixmap = rgb_to_pixmap(rgb)
        self._pixmap_item = pi = scene.addPixmap(pixmap)
        pi.setScale(pixels_per_cell)

    def save_image(self, path):
        size = self.scene.sceneRect().size().toSize()
        from Qt.QtGui import QImage, QPainter
        image = QImage(size, QImage.Format_ARGB32)
        from Qt.QtCore import Qt
        image.fill(Qt.white)
        painter = QPainter(image)
        self.scene.render(painter)
        painter.end()
        image.save(path)
# The following only saves what is visible in the viewport, and includes scrollbars.
#        scene_pixmap = self.grab()
#        scene_pixmap.save(path)

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
