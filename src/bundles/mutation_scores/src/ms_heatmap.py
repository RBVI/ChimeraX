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

    def __init__(self, session, tool_name = 'Mutation Scores Heatmap', name = None, draw = True):

        self.name = name
        self._score_names = []			# Names of shown scores
        self._include_residue_numbers = []	# If empty then include all residues in heatmap
        self._default_amino_acid_order = 'HRKDEFWYNQILCSTVMAGP'
        self._group_spacing = 1	# Number of blank pixels after each amino acid or score group
        self._last_hover_residues = None		# (Residues, res_colors, atom_colors)
        self._last_dragbox_residues = []		# List of (Residues, res_colors, atom_colors)
        self._block_drawing = False
        self._max_axis_font_size = 14
        self._warn_noninteger_cell_size = True

        ToolInstance.__init__(self, session, tool_name)
        self.display_name = tool_name if name is None else f'{tool_name} {name}'

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
        if draw:
            self._draw_graphics()

        # Keep axis labels in view when scrolling heatmap
        vbar, hbar = sv.verticalScrollBar(), sv.horizontalScrollBar()
        vbar.valueChanged.connect(self._viewport_change)
        vbar.rangeChanged.connect(self._viewport_change)
        hbar.valueChanged.connect(self._viewport_change)
        hbar.rangeChanged.connect(self._viewport_change)
        
        tw.manage(placement=None)	# Start floating

    # ---------------------------------------------------------------------------
    #
    def _create_graphics_pane(self, parent):
        gv = ScoreView(parent, self._report_cell_info, self._dragged_box)
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
        if self._block_drawing:
            return
        self._score_view.clear_scene()
        self._x_axis_group = self._y_axis_group = None
        if self._mutation_set is None:
            return
        self._set_heatmap_image()
        self._make_residue_axis_labels()
        if self._grouping == 'amino acid':
            self._make_amino_acid_axis_labels()
        else:
            self._make_score_name_axis_labels()
        scene = self._score_view.scene
        scene.setSceneRect(scene.itemsBoundingRect())
        self._viewport_change()	# Position axes labels at edge of plot

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

        menu.addAction('Save image', self._save_image)

    # ---------------------------------------------------------------------------
    #
    @property
    def _have_structure(self):
        return len(self._mutation_set.associate_chains(self.session)) > 0
    
    # ---------------------------------------------------------------------------
    #
    def _set_heatmap_image(self):
        if self._mutation_set is None or self._block_drawing:
            return
        score_matrix, missing = self._score_matrix()
        colormap = self._colormap()
        rgb = matrix_to_rgb(score_matrix, missing, colormap)
        self._heatmap_height, self._heatmap_width = rgb.shape[0:2]

        # Add divider lines
        group_size = self._num_scores if self._grouping == 'amino acid' else self._num_amino_acids
        divider_line_color = (0,0,0) if group_size > 5 else (255,255,255)
        row_step = group_size + self._group_spacing
        for i in range(group_size,rgb.shape[0],row_step):
            rgb[i:i+self._group_spacing,:,:] = divider_line_color

        if self._gray_missing_structure_residues.enabled:
            mset = self._mutation_set
            if len(mset.associate_chains(self.session)) > 0:
                res, rnums = mset.associated_residues()
                struct_res_nums = set(rnums)
                no_struct_res_indices = [i for i,r in enumerate(self._residue_numbers) if not r in struct_res_nums]
                gray_color = -self._grayout_color.color[:3] + 255
                for i in no_struct_res_indices:
                    rgb[:,i,:] = _shade_gray(rgb[:,i,:], gray_color)

        pixels_per_cell = self._cell_size
        self._score_view.set_image(rgb, pixels_per_cell)

    # ---------------------------------------------------------------------------
    #
    @property
    def _cell_size(self):
        ppc = self._pixels_per_cell.value
        if ppc is None:
            # Warn about non-integer cell size just once.
            if self._warn_noninteger_cell_size:
                self.session.logger.error('Zoom factor must be an integer')
                self._warn_noninteger_cell_size = False
            ppc = 1
        else:
            self._warn_noninteger_cell_size = True
        return max(1, int(ppc))

    # ---------------------------------------------------------------------------
    #
    def _filtered_score_names(self, exclude = ['position'], default_all = True):
        mset = self._mutation_set
        score_names = [score_name for score_name in mset.score_names() if score_name not in exclude]
        sf = self._score_name_filter.value
        snames = [name.strip() for name in sf.split(',')]
        filtered_names = []
        if snames:
            for sname in snames:
                if sname.startswith('*'):
                    suffix = sname[1:]
                    filtered_names.extend([score_name for score_name in score_names if score_name.endswith(suffix)])
                elif sname in score_names:
                    filtered_names.append(sname)
        if len(filtered_names) == 0 and default_all:
            filtered_names = score_names
        return filtered_names

    # ---------------------------------------------------------------------------
    #
    @property
    def _all_score_names(self):
        mset = self._mutation_set
        return mset.score_names() if mset else []
        
    # ---------------------------------------------------------------------------
    #
    @property
    def _amino_acids(self):
        aa = self._amino_acid_order.value.strip()
        if len(aa) == 0:
            aa = self._default_amino_acid_order
        return aa
        
    # ---------------------------------------------------------------------------
    #
    def _score_matrix(self):
        scores = None
        mset = self._mutation_set
        self._score_names = self._filtered_score_names()
        self._num_scores = num_scores = len(self._score_names)
        aa_to_index = {aa:i for i, aa in enumerate(self._amino_acids)}
        self._num_amino_acids = num_aa = len(aa_to_index)
        self._res_aa = res_aa = {}
        subtract_fit = self._subtract_fit
        sub_score_values = mset.score_values(subtract_fit) if subtract_fit else None
        aa_grouping = (self._grouping == 'amino acid')
        group_spacing = 0 if (aa_grouping and num_scores == 1) or (not aa_grouping and num_aa == 1) else 1
        self._group_spacing = group_spacing
        cant_normalize = []
        for snum, score_name in enumerate(self._score_names):
            score_values = mset.score_values(score_name)
            if sub_score_values:
                score_values = score_values.subtract_fit(sub_score_values)
            if scores is None:
                # TODO: Could one score have missing residues that another score includes?
                res_nums = list(score_values.residue_numbers())
                if self._include_residue_numbers:
                    res_nums = [r for r in res_nums if r in self._include_residue_numbers]
                res_nums.sort()
                self._residue_numbers = res_nums
                self._residue_number_to_heatmap_index = resnum_to_index = {r:i for i,r in enumerate(res_nums)}
                num_res = len(res_nums)
                dims = ((num_res, num_aa, num_scores + group_spacing)
                        if aa_grouping else (num_res, num_scores, num_aa + group_spacing))
                from numpy import zeros, ones, float32
                self._scores = scores = zeros(dims, float32)
                self._missing_scores = missing_scores = ones(dims, bool)
            sscores = scores[:,:,snum] if aa_grouping else scores[:,snum,:]
            smissing = missing_scores[:,:,snum] if aa_grouping else missing_scores[:,snum,:]
            for res_num, from_aa, to_aa, value in score_values.all_values():
                if res_num in resnum_to_index:
                    res_aa[res_num] = from_aa
                    if to_aa in aa_to_index:
                        aa_index = aa_to_index[to_aa]
                        r_index = resnum_to_index[res_num]
                        sscores[r_index, aa_index] = value
                        smissing[r_index, aa_index] = False
            if self._normalize.enabled:
                mean, sdev = score_values.synonymous_mean_and_sdev()
                if mean is None or sdev is None or sdev == 0:
                    cant_normalize.append((score_name, sdev))
                else:
                    sscores -= mean
                    sscores /= sdev

        if cant_normalize:
            score_names = ', '.join(score_name for score_name, sdev in cant_normalize)
            sdevs = [sdev for score_name, sdev in cant_normalize]
            if None in sdevs and 0 in sdevs:
                reason = 'there are none or standard deviation is 0'
            elif None in sdevs:
                reason = 'there are none'
            else:
                reason = 'standard deviation is 0'
            self.session.logger.error(f'Cannot normalize {score_names} by synonymous mutation values since {reason}.  Heatmap normalization has been turned off.')
            self._normalize.enabled = False
            return self._score_matrix()

        y_size = scores.shape[1]*scores.shape[2]
        scores_2d = scores.reshape((num_res, y_size)).transpose()
        missing_2d = missing_scores.reshape((num_res, y_size)).transpose()
        if group_spacing > 0:
            scores_2d = scores_2d[:-group_spacing,:]
            missing_2d = missing_2d[:-group_spacing,:]
        return scores_2d, missing_2d

    # ---------------------------------------------------------------------------
    #
    def _report_cell_info(self, column_index, row_index):
        res_num, from_aa, to_aa, score_name, score_value = self._cell_info(column_index, row_index)
        if res_num is None or from_aa is None or score_name is None or to_aa is None:
            msg = ''
        else:
            value = 'missing' if score_value is None else ('%.2g' % score_value)
            msg = f'{from_aa}{res_num}{to_aa} {score_name} {value}'
        self._info_label.setText(msg)

        if self._color_residue_on_hover.enabled and self._have_structure:
            self._uncolor_hover_residues()
            if res_num is not None:
                res, rnums = self._mutation_set.associated_residues([res_num])
                if len(res) > 0:
                    self._last_hover_residues = (res, res.ribbon_colors, res.atoms.colors)
                    color = self._hover_color.color
                    res.ribbon_colors = color
                    res.atoms.colors = color
            
    # ---------------------------------------------------------------------------
    #
    def _cell_info(self, column_index, row_index):
        # Float column index ranges from 0-1 from left to right edge of first pixel.
        c, r = int(round(column_index-0.5)), int(round(row_index-0.5))
        res_num = from_aa = to_aa = score_name = score_value = None
        num_aa = len(self._amino_acids)
        num_scores = self._num_scores
        aa_grouping = (self._grouping == 'amino acid')
        group_size = (num_scores if aa_grouping else num_aa) + self._group_spacing
        num_rows = (num_aa if aa_grouping else num_scores) * group_size
        res_nums = self._residue_numbers
        num_cols = len(res_nums)
        if c >= 0 and c < num_cols and r >= 0 and r < num_rows:
            res_num = res_nums[c]
            from_aa = self._res_aa[res_num]
            score_num = (r % group_size) if aa_grouping else (r // group_size)
            if score_num < num_scores:
                score_name = self._score_names[score_num]
                aa_index = (r // group_size) if aa_grouping else (r % group_size)
                if aa_index < num_aa:
                    to_aa = self._amino_acids[aa_index]
                    if aa_grouping:
                        missing = self._missing_scores[c, aa_index, score_num]
                        score_value = None if missing else self._scores[c, aa_index, score_num]
                    else:
                        missing = self._missing_scores[c, score_num, aa_index]
                        score_value = None if missing else self._scores[c, score_num, aa_index]
        return res_num, from_aa, to_aa, score_name, score_value
        
    # ---------------------------------------------------------------------------
    #
    def _dragged_box(self, xy1, xy2, add = False):
        if not self._drag_color_enabled.value:
            return
        if not self._have_structure:
            return

        (i1,j1),(i2,j2) = xy1,xy2
        w,h = self._heatmap_width, self._heatmap_height
        if (j1 < 0 and j2 < 0) or (j1 >= h and j2 >= h) or (i1 < 0 and i2 < 0) or (i1 >= w and i2 >= w):
            # drag box does not intersect heatmap so clear colors.
            self._uncolor_dragbox_residues()
            return

        all_res_nums = self._residue_numbers
        isize = len(all_res_nums)
        imin, imax = max(0, int(min(i1,i2))), min(isize-1, int(max(i1,i2)))
        if imax < imin:
            return
        res_nums = tuple(all_res_nums[imin:imax+1])
        mset = self._mutation_set
        res, rnums = mset.associated_residues(res_nums)
        if len(res) > 0:
            self._uncolor_hover_residues()	# Avoid remembering hover colored residue
            if not add:
                self._uncolor_dragbox_residues()
            self._last_dragbox_residues.append((res, res.ribbon_colors, res.atoms.colors))
            color = self._dragbox_color.color
            res.ribbon_colors = color
            res.atoms.colors = color

    # ---------------------------------------------------------------------------
    #
    def _uncolor_dragbox_residues(self):
        for last_res, last_res_colors, last_atom_colors in self._last_dragbox_residues[::-1]:
            last_res.ribbon_colors = last_res_colors
            last_res.atoms.colors = last_atom_colors
        self._last_dragbox_residues.clear()

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
    def _save_image(self, *, default_suffix = '_heatmap.png'):
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
            self._full_view_axis_labels()       # Move the axis labels to edges
            self._score_view.save_image(path)
            self._viewport_change()		# Move axis labels back to edge of viewport

    # ---------------------------------------------------------------------------
    #
    def _make_residue_axis_labels(self, residue_step = None):
        if self._label_every_residue.enabled:
            self._make_every_residue_axis_labels()
            return
        pixels_per_cell = self._cell_size
        if residue_step is None:
            if pixels_per_cell == 1:
                residue_step = 100
            elif pixels_per_cell == 2:
                residue_step = 50
            elif pixels_per_cell in (3,4):
                residue_step = 20
            else:
                residue_step = 10
        res_nums = self._residue_numbers
        nres = len(res_nums)
        scene = self._score_view.scene
        labels = []
        i = 0
        while i < nres:
            if i + residue_step//2 >= nres:
                i = nres-1
            r = res_nums[i]
            text = str(r)
            t = scene.addText(text)
            labels.append(t)
            rect = t.boundingRect()
            x = (i+.5)*pixels_per_cell - rect.width()/2
            y = self._heatmap_height * pixels_per_cell
            t.setPos(x, y)
            if i == nres-1:
                break
            i = min(i+residue_step, nres-1)

        self._x_axis_group = self._make_axis_group(labels)

    # ---------------------------------------------------------------------------
    #
    def _make_axis_group(self, labels, y_range = None):
        scene = self._score_view.scene
        xg = scene.createItemGroup(labels)
        from Qt.QtGui import QBrush, QPen
        from Qt.QtCore import Qt
        pen = QPen(Qt.NoPen)	# Don't draw border
        brush = QBrush(Qt.white)
        rect = xg.boundingRect()
        if y_range:
            ymin, ymax = y_range
            rect.setY(ymin)
            rect.setHeight(ymax-ymin)
        backing_rectangle = scene.addRect(rect, pen=pen, brush=brush)
        backing_rectangle.setZValue(-1)
        xg.addToGroup(backing_rectangle)
        return xg
        
    # ---------------------------------------------------------------------------
    #
    def _viewport_change(self):
        if not self.tool_window.ui_area.isVisible():
            # Don't move axes before window is mapped because it
            # then makes the initial window size too small to show full heatmap.
            return	
        sv = self._score_view
        size = sv.viewport().size() 
        p = sv.mapToScene(0, size.height())
        xg = self._x_axis_group
        sr = sv.sceneRect()
        y = p.y() - sr.height()
        xg.setPos(0, min(0,y))
        yg = self._y_axis_group
        x = p.x() - sr.x()
        yg.setPos(max(0,x), 0)

    # ---------------------------------------------------------------------------
    #
    def _full_view_axis_labels(self):
        # Move the axis labels to edges for saving images.
        self._x_axis_group.setPos(0,0)
        self._y_axis_group.setPos(0,0)

    # ---------------------------------------------------------------------------
    #
    def _make_every_residue_axis_labels(self):
        scene = self._score_view.scene
        pixels_per_cell = self._cell_size
        font = self._axis_font(pixels_per_cell)
        font_height = font.pixelSize()
        labels = []
        for i,res_num in enumerate(self._residue_numbers):
            # Place amino acid type in horizontal text
            from_aa = self._res_aa[res_num]
            t = scene.addText(from_aa, font)
            labels.append(t)
            rect = t.boundingRect()
            x = (i+.5)*pixels_per_cell - rect.width()/2
            y = self._heatmap_height * pixels_per_cell
            t.setPos(x, y)
            # Place residue number in vertical text to save space
            t = scene.addText(str(res_num), font)
            labels.append(t)
            rect = t.boundingRect()
            t.setRotation(90)
            x = (i+.5)*pixels_per_cell + rect.height()/2
            y = self._heatmap_height * pixels_per_cell + int(1.5*font_height)
            t.setPos(x, y)

        self._x_axis_group = self._make_axis_group(labels)

    # ---------------------------------------------------------------------------
    #
    def _make_amino_acid_axis_labels(self):
        pixels_per_cell = self._cell_size
        num_scores = self._num_scores
        scores_height = num_scores * pixels_per_cell
        aa_step = (num_scores + self._group_spacing) * pixels_per_cell
        font = self._axis_font(aa_step)
        labels = []
        for i, aa in enumerate(self._amino_acids):
            t = self._score_view.scene.addText(aa, font)
            labels.append(t)
            rect = t.boundingRect()
            x = -rect.width()
            y = i*aa_step + 0.5*scores_height - rect.height()/2
            t.setPos(x, y)

        height = self._heatmap_height * pixels_per_cell
        self._y_axis_group = self._make_axis_group(labels, y_range = (0, height))

    # ---------------------------------------------------------------------------
    #
    def _make_score_name_axis_labels(self):
        pixels_per_cell = self._cell_size
        num_aa = self._num_amino_acids
        aa_height = num_aa * pixels_per_cell
        score_step = (num_aa + self._group_spacing) * pixels_per_cell
        font = self._axis_font(score_step)
        labels = []
        for i, score_name in enumerate(self._score_names):
            t = self._score_view.scene.addText(score_name, font)
            labels.append(t)
            rect = t.boundingRect()
            x = -rect.width()
            y = i*score_step + 0.5*aa_height - rect.height()/2
            t.setPos(x, y)

        height = self._heatmap_height * pixels_per_cell
        self._y_axis_group = self._make_axis_group(labels, y_range = (0, height))

    # ---------------------------------------------------------------------------
    #
    def _axis_font(self, pixel_height):
        scene = self._score_view.scene
        from Qt.QtGui import QFont
        font = QFont(scene.font())
        font.setPixelSize(min(pixel_height, self._max_axis_font_size))
        return font
        
    # ---------------------------------------------------------------------------
    #
    def _create_options_gui(self, parent):
        from chimerax.ui.widgets import CollapsiblePanel
        self._options_panel = p = CollapsiblePanel(parent, title = None, shrink_to_fit = False)
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
        sc = EntriesRow(f, 'Score names', '', ('Choose', self._choose_score_names))
        self._score_name_filter = scf = sc.values[0]
        scf.pixel_width = 250
        scf.return_pressed.connect(self._draw_graphics)

        # Which amino acids.  String of 1-letter codes.
        aao = EntriesRow(f, 'Amino acid order', '')
        self._amino_acid_order = ao = aao.values[0]
        ao.value = self._default_amino_acid_order
        ao.pixel_width = 200
        ao.return_pressed.connect(self._draw_graphics)

        # Grouping on vertical axis.
        group_by_score = (len(msets) > 0 and len(msets[0].score_names()) > 1)
        gp = EntriesRow(f, 'Group by', not group_by_score, 'amino acid', group_by_score, 'score name')
        self._group_amino_acid, self._group_score_name = ga,gs = gp.values
        from chimerax.ui.widgets import radio_buttons
        radio_buttons(ga,gs)
        ga.changed.connect(self._draw_graphics)

        # Residue axis labels
        lr = EntriesRow(f, False, 'Label every residue')
        self._label_every_residue = ler = lr.values[0]
        ler.changed.connect(self._draw_graphics)
        
        # Zoom factor for heatmap
        zf = EntriesRow(f, 'Zoom factor', 2)
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
        nv = EntriesRow(f, 'No value color', ColorButton)
        self._missing_value_color = nvc = nv.values[0]
        nvc.color = 'black'
        nvc.color_changed.connect(self._changed_colormap)
        
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
        hs = EntriesRow(f, False, 'Gray missing structure residues', ColorButton)
        self._gray_missing_structure_residues, self._grayout_color = hs,gc = hs.values
        gc.color = (0.8,0.8,0.8,1.0)	# Gray
        hs.changed.connect(self._set_heatmap_image)
        gc.color_changed.connect(self._set_heatmap_image)

        # Color by dragging box
        dc = EntriesRow(f, True, 'Drag box to color structure', ColorButton, 'linewidth', 3)
        self._drag_color_enabled, self._dragbox_color, self._dragbox_linewidth = e,c,lw = dc.values
        c.color = (1.0,1.0,0,1.0)	# Yellow
        c.color_changed.connect(self._dragbox_color_changed)
        lw.widget.editingFinished.connect(self._dragbox_linewidth_changed)

        # Color residues while hovering.
        hc = EntriesRow(f, False, 'Hover colors structure residue', ColorButton)
        self._color_residue_on_hover, self._hover_color = e,c = hc.values
        c.color = (1.0,1.0,0,1.0)	# Yellow
        e.changed.connect(self._color_on_hover_changed)

        # Action buttons
        from chimerax.ui.widgets import button_row
        br = button_row(f,
                       [('Show selected residues', self._show_only_selected_residues),
                        ('All residues', self._show_all_residues),
                        ('Save image', self._save_image)],
                       spacing = 10)

        return p

    # ---------------------------------------------------------------------------
    #
    def _choose_score_names(self, *, exclude = ['position']):
        all_score_names = [score_name for score_name in self._mutation_set.score_names()
                           if score_name not in exclude]
        current_names = self._filtered_score_names(default_all = False)
        sc = ScoreChooser.get_singleton(self.session)
        sc.show_score_checkbuttons(all_score_names, current_names, self._chose_score_names)
        sc.tool_window.shown = True
        
    # ---------------------------------------------------------------------------
    #
    def _chose_score_names(self, score_names):
        self._score_name_filter.value = ','.join(score_names)
        self._draw_graphics()

    # ---------------------------------------------------------------------------
    #
    def _dragbox_color_changed(self):
        self._score_view.dragbox_color = self._dragbox_color.color[:3]
    def _dragbox_linewidth_changed(self):
        self._score_view.dragbox_linewidth = self._dragbox_linewidth.value

    # ---------------------------------------------------------------------------
    #
    def _color_on_hover_changed(self, enable):
        if not enable:
            self._uncolor_hover_residues()

    # ---------------------------------------------------------------------------
    #
    def _uncolor_hover_residues(self):
        if self._last_hover_residues:
            last_res, last_res_colors, last_atom_colors = self._last_hover_residues
            last_res.ribbon_colors = last_res_colors
            last_res.atoms.colors = last_atom_colors
            self._last_hover_residues = None
        
    # ---------------------------------------------------------------------------
    #
    @property
    def _mutation_set(self):
        mset_name = self._mutation_set_menu.value
        from .ms_data import mutation_scores
        mset = mutation_scores(self.session, mset_name, raise_error = False)
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
    @property
    def _grouping(self):
        return 'amino acid' if self._group_amino_acid.enabled else 'score name'
        
    # ---------------------------------------------------------------------------
    #
    def _show_only_selected_residues(self):
        mset = self._mutation_set
        mset.associate_chains(self.session)
        res, rnums = mset.associated_residues()
        sel_rnums = [rnum for r,rnum in zip(res,rnums) if r.selected]
        if len(sel_rnums) == 0:
            from chimerax.core.errors import UserError
            raise UserError('No mutations for selected residues')
        self._include_residue_numbers = set(sel_rnums)
        self._draw_graphics()
        
    # ---------------------------------------------------------------------------
    #
    def _set_residues(self, residues):
        rset = set(residues)
        mset = self._mutation_set
        mset.associate_chains(self.session)
        res, rnums = mset.associated_residues()
        rnums = [rnum for r,rnum in zip(res,rnums) if r in rset]
        if len(rnums) == 0:
            from chimerax.core.errors import UserError
            raise UserError('No mutations for selected residues')
        self._include_residue_numbers = set(rnums)
        self._draw_graphics()
        
    # ---------------------------------------------------------------------------
    #
    def _show_all_residues(self):
        if self._include_residue_numbers:
            self._include_residue_numbers.clear()
            self._draw_graphics()

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
        no_value_color = rgba8_to_rgba(self._missing_value_color.color)
        colormap = Colormap(values, colors, color_no_value = no_value_color)
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
        return (mean(values), std(values)) if len(values) >= 1 else (0,1)

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
    # Session save and restore.
    #
    SESSION_SAVE = True
    def take_snapshot(self, session, flags):
        data = {'mutation_set_name': self._mutation_set_menu.value,
                'score_names': self._score_name_filter.value,
                'amino_acids': self._amino_acid_order.value,
                'grouping': self._grouping,
                'include_residue_numbers': self._include_residue_numbers,
                'label_every_residue': self._label_every_residue.enabled,
                'pixels_per_cell': self._cell_size,
                'colormaps': self._colormaps,
                'colormap_values': [cv.value for cv in self._colormap_values],
                'colormap_colors': [cc.color for cc in self._colormap_colors],
                'missing_value_color': self._missing_value_color.color,
                'normalize_scores': self._normalize.enabled,
                'subtract_fit': self._use_subtract_fit.enabled,
                'subtract_fit_score_name': self._subtract_score.value,
                'gray_missing': self._gray_missing_structure_residues.value,
                'grayout_color': self._grayout_color.color,
                'drag_to_color': self._drag_color_enabled.value,
                'dragbox_color': self._dragbox_color.color,
                'dragbox_linewidth': self._dragbox_linewidth.value,
                # TODO: Would be nice to restore colored drag boxes.
                'color_residue_on_hover': self._color_residue_on_hover.enabled,
                'hover_color': self._hover_color.color,
                'options_shown': self._options_panel.shown,
                'view_size': (self._score_view.width(), self._score_view.height()),
                'version': '1'}
        return data

    @classmethod
    def restore_snapshot(cls, session, data):
        hm = cls(session)
        hm.configure(data)
        if hm._mutation_set is None:
            hm._after_session_restore = session.ui.timer(0, hm._draw_graphics)
        return hm

    def configure(self, settings):
        self._block_drawing = True
        if 'mutation_set_name' in settings:
            self._mutation_set_menu.value = settings['mutation_set_name']
        if 'score_names' in settings:
            self._score_name_filter.value = settings['score_names']
        if 'amino_acids' in settings:
            self._amino_acid_order.value = settings['amino_acids']
        if 'grouping' in settings:
            if settings['grouping'] == 'amino acid':
                self._group_amino_acid.enabled = True
            else:
                self._group_score_name.enabled = True
        if 'include_residue_numbers' in settings:
            self._include_residue_numbers = settings['include_residue_numbers']
        if 'residues' in settings:
            self._set_residues(settings['residues'])
        if 'label_every_residue' in settings:
            self._label_every_residue.enabled = settings['label_every_residue']
        if 'pixels_per_cell' in settings:
            self._pixels_per_cell.value = settings['pixels_per_cell']
        if 'colormaps' in settings:
            self._colormaps = settings['colormaps']
        if 'normalize_scores' in settings:
            # Change normalize setting before setting colormap
            self._normalize.enabled = settings['normalize_scores']
        if 'colormap_values' in settings:
            for cv, value in zip(self._colormap_values, settings['colormap_values']):
                cv.value = value
        if 'colormap_colors' in settings:
            for cc, color in zip(self._colormap_colors, settings['colormap_colors']):
                cc.color = color
        if 'missing_value_color' in settings:
            self._missing_value_color.color = settings['missing_value_color']
        if 'subtract_fit' in settings:
            self._use_subtract_fit.enabled = settings['subtract_fit']
        if 'subtract_fit_score_name' in settings:
            self._subtract_score.value = settings['subtract_fit_score_name']
        if 'gray_missing' in settings:
            self._gray_missing_structure_residues.value = settings['gray_missing']
        if 'grayout_color' in settings:
            self._grayout_color.color = settings['grayout_color']
        if 'drag_to_color' in settings:
            self._drag_color_enabled.value = settings['drag_to_color']
        if 'dragbox_color' in settings:
            self._dragbox_color.color = settings['dragbox_color']
        if 'dragbox_linewidth' in settings:
            self._dragbox_linewidth.value = settings['dragbox_linewidth']
        if 'color_residue_on_hover' in settings:
            self._color_residue_on_hover.enabled = settings['color_residue_on_hover']
        if 'hover_color' in settings:
            self._hover_color.color = settings['hover_color']
        if 'options_shown' in settings:
            if settings['options_shown'] != self._options_panel.shown:
                self._options_panel.toggle_panel_display()
        if 'view_size' in settings:
            self._score_view._initial_size_hint = settings['view_size']
        self._block_drawing = False
        self._draw_graphics()

class ScoreChooser(ToolInstance):
    help = 'https://www.rbvi.ucsf.edu/chimerax/data/mutation-scores-oct2024/mutation_scores.html'

    def __init__(self, session, tool_name = 'Score Chooser'):
        self._chosen_score_names = []
        self._score_checkbutton_rows = []
        self._max_name_chars_per_line = 50

        ToolInstance.__init__(self, session, tool_name)

        from chimerax.ui import MainToolWindow
        tw = MainToolWindow(self, close_destroys = False)
        self.tool_window = tw
        parent = tw.ui_area

        from chimerax.ui.widgets import vertical_layout
        vertical_layout(parent, margins = (5,0,0,0))

        tw.manage(placement=None)	# Floating

    @classmethod
    def get_singleton(cls, session, create=True):
        from chimerax.core import tools
        return tools.get_singleton(session, cls, 'Score Chooser', create=create)

    def show_score_checkbuttons(self, all_score_names, current_score_names, chosen_callback):
        self._chosen_callback = chosen_callback

        chosen = [score_name for score_name in current_score_names if score_name in all_score_names]
        self._chosen_score_names = chosen
        
        for row in self._score_checkbutton_rows:
            row.frame.deleteLater()
        self._score_checkbutton_rows.clear()

        groups = self._score_name_groups(all_score_names)
        score_names = list(all_score_names) + list(groups.keys())
        checkbuttons = []
        i = 0
        while i < len(score_names):
            row_score_names = []
            line_args = []
            row_chars = 0
            for score_name in score_names[i:]:
                nchar = len(score_name)
                if len(row_score_names) == 0 or row_chars + nchar <= self._max_name_chars_per_line:
                    row_score_names.append(score_name)
                    if score_name in groups:
                        enabled = True
                        for name in groups[score_name]:
                            if name not in chosen:
                                enabled = False
                    else:
                        enabled = (score_name in chosen)
                    line_args.extend([enabled,score_name])
                    row_chars += nchar
                    i += 1
                else:
                    break
            from chimerax.ui.widgets import EntriesRow
            parent = self.tool_window.ui_area
            score_checkbuttons = EntriesRow(parent, *line_args)
            checkbuttons.extend(score_checkbuttons.values)
            self._score_checkbutton_rows.append(score_checkbuttons)
            for score_name, checkbutton in zip(row_score_names,score_checkbuttons.values):
                names = groups.get(score_name, [score_name])
                checkbutton.widget.clicked.connect(lambda enabled, names=names:
                                                   self._score_chosen(names, enabled))

        self._checkbuttons = {score_name:checkbutton for score_name, checkbutton in zip(score_names, checkbuttons)}

    def _score_chosen(self, score_names, enabled):
        chosen = self._chosen_score_names
        if enabled:
            for score_name in score_names:
                if score_name not in chosen:
                    chosen.append(score_name)
        else:
            for score_name in score_names:
                if score_name in chosen:
                    chosen.remove(score_name)
        # Make "all_*" set all the individual checkbutton names
        for score_name in score_names:
            self._checkbuttons[score_name].value = enabled
        self._chosen_callback(chosen)

    def _score_name_groups(self, all_score_names, suffix_separator = '_'):
        groups = {}	# Map suffix to list of score names.
        for score_name in all_score_names:
            if suffix_separator in score_name:
                suffix = score_name.rsplit(suffix_separator, maxsplit = 1)[1]
                if suffix in groups:
                    groups[suffix].append(score_name)
                else:
                    groups[suffix] = [score_name]
        large_groups = {f'all_{suffix}': score_names
                        for suffix, score_names in groups.items()
                        if len(score_names) >= 3}
        return large_groups

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
def _shade_gray(rgb_array, gray):
    from numpy import clip, int32
    return clip(rgb_array.astype(int32) - gray, 0, 255).astype(rgb_array.dtype)
    
# ---------------------------------------------------------------------------
#
from Qt.QtWidgets import QGraphicsView
class ScoreView(QGraphicsView):
    def __init__(self, parent, report_cell_info_cb=None, rectangle_select_cb=None):
        QGraphicsView.__init__(self, parent)
        self._report_cell_info_callback = report_cell_info_cb
        self._pixmap_item = None
        self._mouse_down = False
        self._shift_mod = False
        self._drag_boxes = []
        self._new_dragbox = False
        self._down_xy = None
        self._rectangle_select_callback = rectangle_select_cb
        self.dragbox_color = (255,255,0)
        self.dragbox_linewidth = 3
        self._initial_size_hint = None

        from Qt.QtWidgets import QGraphicsScene
        self.scene = gs = QGraphicsScene(self)
        self.setScene(gs)

        # Report cell info as mouse hovers over plot.
        if report_cell_info_cb:
            self.setMouseTracking(True)

    def sizeHint(self):
        if self._initial_size_hint is None:
            rect = self.scene.itemsBoundingRect()
            size = rect.size().toSize()
        else:
            w,h = self._initial_size_hint
            from Qt.QtCore import QSize
            size = QSize(w,h)
        return size

    def mousePressEvent(self, event):
        if self._pixmap_item is None:
            return
        from Qt.QtCore import Qt
        self._shift_mod = shift = (event.modifiers() == Qt.KeyboardModifier.ShiftModifier)
        self._mouse_down = True
        self._down_xy = self._scene_position(event)
        self._new_dragbox = True
        if not shift:
            self._clear_drag_boxes()

    def mouseMoveEvent(self, event):
        if self._report_cell_info_callback is None:
            return
        if self._pixmap_item is None:
            return
        # This gives event handler is for QAbstractScrollArea and the event is in viewport() coordinates.
        # We need the event in QGraphicsView coordinates which differ by 1 pixel in x,y.
        gv_point = self.mapFromGlobal(event.globalPosition())  # Map from viewport to graphics view.
        x,y = self.graphics_view_to_image_position(gv_point)
        self._report_cell_info_callback(x,y)

        if self._mouse_down:
            self._drag(event)

    def _drag(self, event):
        self._draw_drag_box(event)

    def mouseReleaseEvent(self, event):
        if self._pixmap_item is None:
            return
        if self._mouse_down:
            self._mouse_down = False
            self._drag(event)
            if self._rectangle_select_callback and self._down_xy:
                corner1 = self._scene_to_image(*self._down_xy)
                up_xy = self._scene_position(event)
                corner2 = self._scene_to_image(*up_xy)
                self._rectangle_select_callback(corner1, corner2, add = self._shift_mod)

    def _scene_position(self, event):
        p = self.mapToScene(event.pos())
        return p.x(), p.y()

    def _scene_to_image(self, scene_x, scene_y):
        ip = self._pixmap_item.mapFromScene(scene_x, scene_y)
        return ip.x(), ip.y()

    def _draw_drag_box(self, event):
        x1,y1 = self._down_xy
        x2,y2 = self._scene_position(event)
        x,y,w,h = min(x1,x2), min(y1,y2), abs(x2-x1), abs(y2-y1)

        if self._new_dragbox:
            from Qt.QtGui import QPen, QColor
            pen = QPen()
            pen.setColor(QColor(*self.dragbox_color))
            pen.setWidth(self.dragbox_linewidth)
            rect = self.scene.addRect(x,y,w,h, pen=pen)
            self._drag_boxes.append(rect)
            self._new_dragbox = False
        else:
            rect = self._drag_boxes[-1]
            rect.setRect(x,y,w,h)

    def _clear_drag_boxes(self):
        boxes = self._drag_boxes
        if boxes:
            for box in boxes:
                self.scene.removeItem(box)
            boxes.clear()

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
        pi.setZValue(-10)	# Put pixmap below axis labels.

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

    def clear_scene(self):
        self.scene.clear()
        self._pixmap_item = None
        self._drag_boxes.clear()

# -----------------------------------------------------------------------------
#
def matrix_to_rgb(matrix, missing, colormap):
    rgb_flat = colormap.interpolated_rgba8(matrix.ravel())[:,:3]
    if missing.any():
        if colormap.color_no_value is None:
            no_value_color = (0,0,0)
        else:
            from chimerax.core.colors import rgba_to_rgba8
            no_value_color = rgba_to_rgba8(colormap.color_no_value)[:3]
        rgb_flat[missing.ravel()] = no_value_color
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

# -----------------------------------------------------------------------------
#
def mutation_heatmap(session, heatmap_name = None,
                     mutation_set = None, scores = None, amino_acids = None,
                     grouping = None, residues = None, label_every_residue = None,
                     pixels_per_cell = None, palette = None, missing_value_color = None,
                     normalize_scores = None, subtract_fit = None, gray_missing = None,
                     grayout_color = None, drag_to_color = None, dragbox_color = None,
                     dragbox_linewidth = None, color_residue_on_hover = None, hover_color = None,
                     show_options = None, size = None, save_image = None):
    settings = {}
    if mutation_set is not None:
        from .ms_data import mutation_scores
        mset = mutation_scores(session, mutation_set)  # Raises error if name not found
        settings['mutation_set_name'] = mutation_set
    if scores is not None:
        settings['score_names'] = scores
    if amino_acids is not None:
        settings['amino_acids'] = amino_acids
    if grouping is not None:
        settings['grouping'] = grouping
    if residues is not None:
        settings['residues'] = residues
    if label_every_residue is not None:
        settings['label_every_residue'] = label_every_residue
    if pixels_per_cell is not None:
       settings['pixels_per_cell'] = pixels_per_cell
    if palette is not None:
        if len(palette.colors) != 4 or not (palette.colors[1] == palette.colors[2]).all():
            from chimerax.core.errors import UserError
            raise UserError('Heatmaps requires a palette with 4 colors where the middle two colors are the same')
        if palette.values_specified:
            settings['colormap_values'] = palette.data_values
        from chimerax.core.colors import rgba_to_rgba8
        settings['colormap_colors'] = [rgba_to_rgba8(color) for color in palette.colors]
    if missing_value_color is not None:
        settings['missing_value_color'] = missing_value_color
    if normalize_scores is not None:
        settings['normalize_scores'] = normalize_scores
    if subtract_fit is not None:
        settings['subtract_fit'] = True
        settings['subtract_fit_score_name'] = subtract_fit
    if gray_missing is not None:
        settings['gray_missing'] = gray_missing
    if grayout_color is not None:
        settings['grayout_color'] = grayout_color
    if drag_to_color is not None:
        settings['drag_to_color'] = drag_to_color
    if dragbox_color is not None:
        settings['dragbox_color'] = dragbox_color
    if dragbox_linewidth is not None:
        settings['dragbox_linewidth'] = dragbox_linewidth
    if color_residue_on_hover is not None:
        settings['color_residue_on_hover'] = color_residue_on_hover
    if hover_color is not None:
        settings['hover_color'] = hover_color
    if show_options is not None:
        settings['options_shown'] = show_options
    if size is not None:
        settings['view_size'] = size

    hm = None
    if heatmap_name is not None:
        hm = _find_named_heatmap(session, heatmap_name)
    if hm is None:
        if heatmap_name is None:
            heatmap_name = _next_heatmap_name(session)
        hm = MutationScoresHeatmap(session, name = heatmap_name, draw = False)

    hm.configure(settings)

    if save_image is not None:
        hm._score_view.save_image(save_image)

    return hm

# -----------------------------------------------------------------------------
#
def _find_named_heatmap(session, heatmap_name):
    for tool in session.tools:
        if isinstance(tool, MutationScoresHeatmap) and tool.name == heatmap_name:
            return tool
    return None

# -----------------------------------------------------------------------------
#
def _next_heatmap_name(session):
    names = set(tool.name for tool in session.tools if isinstance(tool, MutationScoresHeatmap) and tool.name)
    i = 1
    while str(i) in names:
        i += 1
    return str(i)

# -----------------------------------------------------------------------------
#
def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, StringArg, BoolArg, EnumOf, IntArg, Int2Arg
    from chimerax.core.commands import ColormapArg, Color8Arg, SaveFileNameArg
    from chimerax.atomic import ResiduesArg
    desc = CmdDesc(
        optional = [('heatmap_name', StringArg)],
        keyword = [('mutation_set', StringArg),
                   ('scores', StringArg),
                   ('amino_acids', StringArg),
                   ('grouping', EnumOf(['amino acid', 'score'])),
                   ('residues', ResiduesArg),
                   ('label_every_residue', BoolArg),
                   ('pixels_per_cell', IntArg),
                   ('palette', ColormapArg),
                   ('missing_value_color', Color8Arg),
                   ('normalize_scores', BoolArg),
                   ('subtract_fit', StringArg),
                   ('gray_missing', BoolArg),
                   ('grayout_color', Color8Arg),
                   ('drag_to_color', BoolArg),
                   ('dragbox_color', Color8Arg),
                   ('dragbox_linewidth', IntArg),
                   ('color_residue_on_hover', BoolArg),
                   ('hover_color', Color8Arg),
                   ('show_options', BoolArg),
                   ('size', Int2Arg),
                   ('save_image', SaveFileNameArg),
                   ],
        synopsis = 'Show a heatmap of mutation scores.'
    )
    register('mutationscores heatmap', desc, mutation_heatmap, logger=logger)
