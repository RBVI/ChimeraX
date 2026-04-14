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

        self._include_residue_numbers = []	# If empty then include all residues in heatmap
        self._default_amino_acid_order = 'HRKDEFWYNQILCSTVMAGP'
        self._group_spacing = 1	# Number of blank pixels after each amino acid or score group
        self._last_hover_residues = None		# (Residues, res_colors, atom_colors)
        self._last_dragbox_residues = []		# List of (Residues, res_colors, atom_colors)
        
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
        self._score_view.clear_scene()
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
        if self._mutation_set is None:
            return
        score_matrix = self._score_matrix()
        colormap = self._colormap()
        rgb = matrix_to_rgb(score_matrix, colormap)
        self._heatmap_height = rgb.shape[0]

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
        self._num_scores = num_scores = len(self._score_names)
        aa_to_index = {aa:i for i, aa in enumerate(self._amino_acids)}
        self._num_amino_acids = num_aa = len(aa_to_index)
        self._res_aa = res_aa = {}
        subtract_fit = self._subtract_fit
        sub_score_values = mset.score_values(subtract_fit) if subtract_fit else None
        aa_grouping = (self._grouping == 'amino acid')
        group_spacing = 0 if (aa_grouping and num_scores == 1) or (not aa_grouping and num_aa == 1) else 1
        self._group_spacing = group_spacing
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
                dims = ((num_aa, num_scores + group_spacing)
                        if aa_grouping else (num_scores, num_aa + group_spacing))
                from numpy import zeros, float32
                self._scores = scores = zeros((num_res,)+dims, float32)
            sscores = scores[:,:,snum] if aa_grouping else scores[:,snum,:]
            for res_num, from_aa, to_aa, value in score_values.all_values():
                if res_num in resnum_to_index:
                    res_aa[res_num] = from_aa
                    if to_aa in aa_to_index:
                        aa_index = aa_to_index[to_aa]
                        r_index = resnum_to_index[res_num]
                        sscores[r_index, aa_index] = value
            if self._normalize.enabled:
                mean, sdev = score_values.synonymous_mean_and_sdev()
                sscores -= mean
                sscores /= sdev

        y_size = scores.shape[1]*scores.shape[2]
        scores_2d = scores.reshape((num_res, y_size)).transpose()
        if group_spacing > 0:
            scores_2d = scores_2d[:-group_spacing,:]
        return scores_2d

    # ---------------------------------------------------------------------------
    #
    def _report_cell_info(self, column_index, row_index):
        res_num, from_aa, to_aa, score_name, score_value = self._cell_info(column_index, row_index)
        if score_value is None:
            msg = ''
        else:
            msg = f'{from_aa}{res_num}{to_aa} {score_name} {"%.2f"%score_value}'
        self._info_label.setText(msg)

        if res_num is not None and self._color_residue_on_hover.enabled and self._have_structure:
            self._uncolor_hover_residues()
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
                        score_value = self._scores[c, aa_index, score_num]
                    else:
                        score_value = self._scores[c, score_num, aa_index]
        return res_num, from_aa, to_aa, score_name, score_value
        
    # ---------------------------------------------------------------------------
    #
    def _dragged_box(self, xy1, xy2, add = False):
        if not self._drag_color_enabled.value:
            return
        if not self._have_structure:
            return
        i1,i2 = xy1[0],xy2[0]
        all_res_nums = self._residue_numbers
        isize = len(all_res_nums)
        imin, imax = max(0, int(min(i1,i2))), min(isize-1, int(max(i1,i2)))
        if imax < imin:
            return
        if not add:
            self._uncolor_dragbox_residues()
        res_nums = tuple(all_res_nums[imin:imax+1])
        mset = self._mutation_set
        res, rnums = mset.associated_residues(res_nums)
        if len(res) > 0:
            self._last_dragbox_residues.append((res, res.ribbon_colors, res.atoms.colors))
            color = self._dragbox_color.color
            res.ribbon_colors = color
            res.atoms.colors = color

    # ---------------------------------------------------------------------------
    #
    def _uncolor_dragbox_residues(self):
        for last_res, last_res_colors, last_atom_colors in self._last_dragbox_residues:
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
            self._score_view.save_image(path)

    # ---------------------------------------------------------------------------
    #
    def _make_residue_axis_labels(self, residue_step = None):
        if self._label_every_residue.enabled:
            self._make_every_residue_axis_labels()
            return
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
        res_nums = self._residue_numbers
        iranges = _contiguous_ranges(res_nums)
        rpad = residue_step // 2
        for imin, imax in iranges:
            rmin, rmax = res_nums[imin], res_nums[imax]
            smin, smax = ((rmin+rpad)//residue_step) + 1, (rmax-(rpad+1))//residue_step
            rnums = [rmin] if rmax == rmin else [rmin] + [s*residue_step for s in range(smin, smax+1)] + [rmax]
            for r in rnums: 
                text = str(r)
                t = self._score_view.scene.addText(text)
                rect = t.boundingRect()
                x = (r-rmin+imin+.5)*pixels_per_cell - rect.width()/2
                y = self._heatmap_height * pixels_per_cell
                t.setPos(x, y)

    # ---------------------------------------------------------------------------
    #
    def _make_every_residue_axis_labels(self):
        scene = self._score_view.scene
        pixels_per_cell = self._pixels_per_cell.value
        for i,res_num in enumerate(self._residue_numbers):
            from_aa = self._res_aa[res_num]
            text = f'{from_aa}{res_num}'
            t = scene.addText(text)
            rect = t.boundingRect()
            t.setRotation(90)
            x = (i+.5)*pixels_per_cell + rect.height()/2
            y = self._heatmap_height * pixels_per_cell
            t.setPos(x, y)

    # ---------------------------------------------------------------------------
    #
    def _make_amino_acid_axis_labels(self):
        pixels_per_cell = self._pixels_per_cell.value
        num_scores = self._num_scores
        scores_height = num_scores * pixels_per_cell
        aa_step = (num_scores + self._group_spacing) * pixels_per_cell
        for i, aa in enumerate(self._amino_acids):
            t = self._score_view.scene.addText(aa)
            rect = t.boundingRect()
            x = -rect.width()
            y = i*aa_step + 0.5*scores_height - rect.height()/2
            t.setPos(x, y)

    # ---------------------------------------------------------------------------
    #
    def _make_score_name_axis_labels(self):
        pixels_per_cell = self._pixels_per_cell.value
        num_aa = self._num_amino_acids
        aa_height = num_aa * pixels_per_cell
        score_step = (num_aa + self._group_spacing) * pixels_per_cell
        for i, score_name in enumerate(self._score_names):
            t = self._score_view.scene.addText(score_name)
            rect = t.boundingRect()
            x = -rect.width()
            y = i*score_step + 0.5*aa_height - rect.height()/2
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

        # Which amino acids.  String of 1-letter codes.
        aao = EntriesRow(f, 'Amino acid order', '')
        self._amino_acid_order = ao = aao.values[0]
        ao.value = self._default_amino_acid_order
        ao.pixel_width = 200
        ao.return_pressed.connect(self._draw_graphics)

        # Grouping on vertical axis.
        gp = EntriesRow(f, 'Group by', True, 'amino acid', False, 'score name')
        self._group_amino_acid, self._group_score_name = ga,gs = gp.values
        from chimerax.ui.widgets import radio_buttons
        radio_buttons(ga,gs)
        ga.changed.connect(self._draw_graphics)

        # Residue axis labels
        lr = EntriesRow(f, False, 'Label every residue')
        self._label_every_residue = ler = lr.values[0]
        ler.changed.connect(self._draw_graphics)
        
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
                'pixels_per_cell': self._pixels_per_cell.value,
                'colormaps': self._colormaps,
                'colormap_values': [cv.value for cv in self._colormap_values],
                'colormap_colors': [cc.color for cc in self._colormap_colors],
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
        hm._mutation_set_menu.value = data['mutation_set_name']
        hm._score_name_filter.value = data['score_names']
        hm._amino_acid_order.value = data['amino_acids']
        if data['grouping'] == 'amino acid':
            hm._group_amino_acid.enabled = True
        else:
            hm._group_score_name.enabled = True
        hm._include_residue_numbers = data['include_residue_numbers']
        hm._label_every_residue.enabled = data['label_every_residue']
        hm._pixels_per_cell.value = data['pixels_per_cell']
        hm._colormaps = data['colormaps']
        for cv, value in zip(hm._colormap_values, data['colormap_values']):
            cv.value = value
        for cc, color in zip(hm._colormap_colors, data['colormap_colors']):
            cc.color = color
        hm._normalize.enabled = data['normalize_scores']
        hm._use_subtract_fit.enabled = data['subtract_fit']
        hm._subtract_score.value = data['subtract_fit_score_name']
        hm._gray_missing_structure_residues.value = data['gray_missing']
        hm._grayout_color.color = data['grayout_color']
        hm._drag_color_enabled.value = data['drag_to_color']
        hm._dragbox_color.color = data['dragbox_color']
        hm._dragbox_linewidth.value = data['dragbox_linewidth']
        hm._color_residue_on_hover.enabled = data['color_residue_on_hover']
        hm._hover_color.color = data['hover_color']
        if data['options_shown']:
            hm._options_panel.toggle_panel_display()
        hm._score_view._initial_size_hint = data['view_size']
        if hm._mutation_set is None:
            hm._after_session_restore = session.ui.timer(0, hm._draw_graphics)
        print ('heatmap restore data', data)
        return hm

# ---------------------------------------------------------------------------
#
def _contiguous_ranges(int_array):
    ranges = []
    vprev = None
    istart = 0
    for i,v in enumerate(int_array):
        if vprev is not None and v != vprev+1:
            ranges.append((istart,i-1))
            istart = i
        vprev = v
    ranges.append((istart,i))
    return ranges

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
