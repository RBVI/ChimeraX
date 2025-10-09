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

# Plot a histogram of mutation scores.
def mutation_scores_histogram(session, score_name, mutation_set = None,
                              bins = 20, curve = True, smooth_width = None, smooth_bins = 200,
                              scale = 'log', synonymous = True, bounds = True, replace = True):
    plot = _find_mutation_histogram(session, mutation_set) if replace else None
    if plot is None:
        plot = MutationHistogram(session)

    values = plot.set_plot_data(score_name, mutation_set, bins=bins, scale=scale,
                                curve=curve, smooth_width=smooth_width, smooth_bins=smooth_bins,
                                synonymous = synonymous, bounds = bounds)

    range = f'having range {"%.3g"%min(values)} to {"%.3g"%max(values)}' if len(values) > 0 else ''
    message = f'Plotted {len(values)} scores {range} for {score_name}'
    session.logger.info(message)

from chimerax.interfaces.graph import Graph
class MutationHistogram(Graph):

    def __init__(self, session):
        self.mutation_set_name = ''
        self._synonymous_histogram = None
        self._bounds_artists = None
        self._drag_colors_structure = True
        
        nodes = edges = []
        Graph.__init__(self, session, nodes, edges,
                       tool_name = 'Mutation scores histogram', title = 'Mutation scores histogram',
                       hide_ticks = False, drag_select_callback = self._rectangle_selected,
                       zoom_axes = 'x', translate_axes = 'x')

        tw = self.tool_window
        parent = tw.ui_area
        layout = parent.layout()

        # Add score and mutation set menus.
        from chimerax.ui.widgets import EntriesRow
        menus = EntriesRow(parent, ' Score', ('score1', 'score2'), 'Mutations', ('set1', 'set2'))
        self._score_menu, self._mutation_set_menu = menus.values
        for m in menus.values:
            menu = m.widget.menu()
            menu.aboutToShow.connect(lambda menu=menu: self._menu_about_to_show(menu))
            menu.triggered.connect(self._menu_selection_changed)
        layout.addWidget(menus.frame)

    def _menu_about_to_show(self, menu):
        menu.clear()
        if menu is self._mutation_set_menu.widget.menu():
            from .ms_data import mutation_scores_list
            for ms_name in mutation_scores_list(self.session):
                menu.addAction(ms_name)
        else:
            from .ms_data import mutation_scores
            ms_name = self._mutation_set_menu.value
            mset = mutation_scores(self.session, ms_name)
            for name in mset.score_names():
                menu.addAction(name)

    def _menu_selection_changed(self, action):
        mutation_set_name = self._mutation_set_menu.value
        score_name = self._score_menu.value

        if mutation_set_name != self.mutation_set_name:
            # If mutation set is changed, make sure the score name s valid.
            from .ms_data import mutation_scores
            mset = mutation_scores(self.session, mutation_set_name)
            score_names = mset.score_names()
            if score_name not in score_names:
                score_name = self._score_menu.value = score_names[0]
            
        scale = self._yscale

        self.set_plot_data(score_name, mutation_set_name, bins = self._bins, scale = self._yscale,
                           curve = self._smooth_curve, smooth_width = self._smooth_width, smooth_bins = self._smooth_bins,
                           synonymous = self._show_synonymous, bounds = self._synonymous_bounds)

    def set_plot_data(self, score_name, mutation_set_name, bins = 20, scale = 'linear',
                      curve = True, smooth_bins = 20, smooth_width = None,
                      synonymous = True, bounds = True):
        from .ms_data import mutation_scores
        mset = mutation_scores(self.session, mutation_set_name)
        score_values = mset.score_values(score_name)

        self._score_menu.value = score_name
        self._mutation_set_menu.value = self.mutation_set_name = mset.name

        values = [value for res_num, from_aa, to_aa, value in score_values.all_values()]
        syn_values = [value for res_num, from_aa, to_aa, value in score_values.all_values() if to_aa == from_aa]

        self._set_values(values, title=mset.name, x_label=score_name, bins=bins, yscale=scale,
                         smooth_curve=curve, smooth_width=smooth_width, smooth_bins=smooth_bins,
                         show_synonymous = synonymous, synonymous_scores = syn_values, synonymous_bounds = bounds)
        return values
    
    def _set_values(self, scores, title = '', x_label = '', bins = 20, yscale = 'log',
                    smooth_curve = False, smooth_width = None, smooth_bins = 200,
                    show_synonymous = False, synonymous_scores = None, synonymous_bounds = False):
        # Remember state for session saving
        self._scores, self._bins, self._smooth_curve, self._smooth_width, self._smooth_bins = \
            scores, bins, smooth_curve, smooth_width, smooth_bins
        self._show_synonymous, self._synonymous_scores, self._synonymous_bounds = \
            show_synonymous, synonymous_scores, synonymous_bounds

        a = self.axes
        a.clear()
        self._synonymous_histogram = self._bounds_artists = None

        a.hist(scores, bins=bins, color = 'lightgray')

        self._show_synonymous_histogram(show_synonymous)
        self._show_synonymous_bounds(synonymous_bounds)

        a.set_title(title)
        a.set_xlabel(x_label)
        a.set_ylabel('Count')
        if smooth_curve:
            x, y = gaussian_histogram(scores, sdev = smooth_width, bins = smooth_bins)
            y *= (smooth_bins / bins)	# Scale to match histogram bar height
            a.plot(x, y, color = 'orange')

        self.canvas.draw()
        ymin, self._ymax = a.get_ylim()
        if yscale == 'log':
            self._set_log_scale()

    def tight_layout(self):
        # Don't hide axes and reduce padding
        pass

    @property
    def mutation_set(self):
        from .ms_data import mutation_scores
        mset = mutation_scores(self.session, self.mutation_set_name)
        return mset

    def _rectangle_selected(self, event1, event2):
        x1, x2 = event1.xdata, event2.xdata
        xmin, xmax = min(x1,x2), max(x1,x2)
        mset = self.mutation_set
        if mset is None:
            return
        score_name = self._score_menu.value
        score_values = mset.score_values(score_name)
        res_nums = set([res_num for res_num, from_aa, to_aa, value in score_values.all_values()
                        if value >= xmin and value <= xmax])
        mset.associate_chains(self.session)
        res, rnums = mset.associated_residues(res_nums)

        if len(res) > 0:
            from chimerax.atomic import concise_residue_spec
            rspec = concise_residue_spec(self.session, res)
            cmds = [f'select {rspec}']
            if self._drag_colors_structure:
                from chimerax.atomic import concise_chain_spec
                cspec = concise_chain_spec(res.unique_chains)
                cmds.append(f'color {cspec} lightgray ; color {rspec} lime')
        else:
            cmds = ['select clear']
            if self._drag_colors_structure:
                chains = mset.associated_chains()
                if len(chains) > 0:
                    from chimerax.atomic import concise_chain_spec
                    cspec = concise_chain_spec(chains)
                    cmds.append(f'color {cspec} lightgray')
        for cmd in cmds:
            self._run_command(cmd)

    def _fill_context_menu(self, menu, x, y):
        if self._yscale == 'linear':
            self.add_menu_entry(menu, 'Switch linear to log scale', self._set_log_scale)
        else:
            self.add_menu_entry(menu, 'Switch log to linear scale', self._set_linear_scale)

        show_syn = not self._show_synonymous
        show_or_hide = 'Show' if show_syn else 'Hide'
        self.add_menu_entry(menu, f'{show_or_hide} synonymous',
                            lambda show_syn=show_syn: self._show_synonymous_histogram(show_syn))
        show_bounds = not self._synonymous_bounds
        show_or_hide = 'Show' if show_bounds else 'Hide'
        self.add_menu_entry(menu, f'{show_or_hide} synonymous bounds',
                            lambda show_bounds=show_bounds: self._show_synonymous_bounds(show_bounds))
        a = self.add_menu_entry(menu, f'Ctrl-drag colors structure', self._toggle_drag_colors_structure)
        a.setCheckable(True)
        a.setChecked(self._drag_colors_structure)
            
        self.add_menu_separator(menu)
        self.add_menu_entry(menu, 'New histogram', self._copy_histogram)
        self.add_menu_entry(menu, 'Save plot as...', self.save_plot_as)

    @property
    def _yscale(self):
        return self.axes.get_yscale()
    def _set_log_scale(self):
        a = self.axes
        a.set_yscale('log')
        a.set_ylim(1)
        self.canvas.draw()
    def _set_linear_scale(self):
        a = self.axes
        a.set_yscale('linear')
        a.set_ylim(0, self._ymax)
        self.canvas.draw()

    def _show_synonymous_histogram(self, show = True):
        if show and self._synonymous_histogram is None and self._synonymous_scores:
            _,_,self._synonymous_histogram = \
                self.axes.hist(self._synonymous_scores, bins = self._bins, color = 'blue')
        elif not show and self._synonymous_histogram:
            self._synonymous_histogram.remove()
            self._synonymous_histogram = None
        self._show_synonymous = show
        self.canvas.draw()
    def _show_synonymous_bounds(self, show = True):
        if show and not self._bounds_artists and self._synonymous_scores:
            from numpy import mean, std
            m, d = mean(self._synonymous_scores), std(self._synonymous_scores)
            sd = standard_deviations = 2
            smin, smax = m - sd*d, m + sd*d
            a = self.axes
            ymin, ymax = a.get_ylim()
            segment_xy = [((smin,smin), (ymin,ymax)), ((smax,smax), (ymin,ymax))]
            from matplotlib.lines import Line2D
            lines = [Line2D(xdata, ydata, color = 'black', linestyle = 'dotted', zorder = 10)
                     for xdata, ydata in segment_xy]
            self._bounds_artists = [a.add_artist(line) for line in lines]
        elif not show and self._bounds_artists:
            for ba in self._bounds_artists:
                if ba.axes is not None:
                    ba.remove()
            self._bounds_artists.clear()
        self._synonymous_bounds = show
        self.canvas.draw()

    def _toggle_drag_colors_structure(self):
        self._drag_colors_structure = not self._drag_colors_structure

    def _run_command(self, command):
        from chimerax.core.commands import run
        run(self.session, command)

    def _copy_histogram(self):
        copy = MutationHistogram(self.session)
        copy.set_plot_data(self._score_menu.value, self._mutation_set_menu.value, bins = self._bins,
                           curve = self._smooth_curve, smooth_width = self._smooth_width, smooth_bins = self._smooth_bins,
                           scale = self._yscale)
        return copy

    # ---------------------------------------------------------------------------
    # Session save and restore.
    #
    SESSION_SAVE = True
    def take_snapshot(self, session, flags):
        axes = self.axes
        data = {'mutation_set_name': self.mutation_set_name,
                'score_name': self._score_menu.value,
                'scores': self._scores,
                'title': axes.get_title(),
                'x_label': axes.get_xlabel(),
                'bins': self._bins,
                'smooth_curve': self._smooth_curve,
                'smooth_width': self._smooth_width,
                'smooth_bins': self._smooth_bins,
                'yscale': self._yscale,
                'show_synonymous': self._show_synonymous,
                'synonymous_scores': self._synonymous_scores,
                'synonymous_bounds': self._synonymous_bounds,
                'version': '1'}
        return data

    @classmethod
    def restore_snapshot(cls, session, data):
        hp = cls(session)
        hp.mutation_set_name = hp._mutation_set_menu.value = data['mutation_set_name']
        hp._score_menu.value = data.get('score_name', '')
        hp._set_values(data['scores'], title = data['title'], x_label = data['x_label'], bins = data['bins'],
                       smooth_curve = data['smooth_curve'], smooth_width = data['smooth_width'],
                       smooth_bins = data['smooth_bins'], yscale = data.get('yscale', 'linear'),
                       show_synonymous = data.get('show_synonymous'),
                       synonymous_scores = data.get('synonymous_scores'), synonymous_bounds = data.get('synonymous_bounds'))
        return hp

def gaussian_histogram(values, sdev = None, pad = 5, bins = 256):
    '''Make a smooth curve approximating a histogram by convolution with a Gaussian.'''
    if sdev is None:
        from numpy import std
        sdev = 0.1 * std(values)
    from numpy import max, min
    vmin, vmax = min(values), max(values)
    hrange = (vmin - pad*sdev, vmax + pad*sdev)
    from numpy import histogram, float32
    hist, bin_edges = histogram(values, bins, hrange)
    
    ijk_sdev = (0, 0, bins * sdev / (hrange[1] - hrange[0]))
    from chimerax.map_filter.gaussian import gaussian_convolution
    y = gaussian_convolution(hist.reshape((bins,1,1)).astype(float32), ijk_sdev).reshape((bins,))
    x = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    return x, y

def _find_mutation_histogram(session, mutation_set_name):
    hists = [tool for tool in session.tools.list()
             if isinstance(tool, MutationHistogram) and (mutation_set_name is None or tool.mutation_set_name == mutation_set_name)]
    return hists[-1] if hists else None

def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, StringArg, BoolArg, FloatArg, IntArg, EnumOf
    desc = CmdDesc(
        required = [('score_name', StringArg)],
        keyword = [('mutation_set', StringArg),
                   ('bins', IntArg),
                   ('scale', EnumOf(['linear', 'log'])),
                   ('curve', BoolArg),
                   ('smooth_width', FloatArg),
                   ('smooth_bins', IntArg),
                   ('synonymous', BoolArg),
                   ('bounds', BoolArg),
                   ('replace', BoolArg),
                   ],
        synopsis = 'Show histogram of mutation scores'
    )
    register('mutationscores histogram', desc, mutation_scores_histogram, logger=logger)
