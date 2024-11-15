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
                              scale = 'linear', replace = True):
    plot = _find_mutation_histogram(session, mutation_set) if replace else None
    if plot is None:
        plot = MutationHistogram(session)

    values = plot.set_plot_data(score_name, mutation_set, bins=bins,
                                curve=curve, smooth_width=smooth_width, smooth_bins=smooth_bins,
                                scale=scale)

    range = f'having range {"%.3g"%min(values)} to {"%.3g"%max(values)}' if len(values) > 0 else ''
    message = f'Plotted {len(values)} scores {range} for {score_name}'
    session.logger.info(message)

from chimerax.interfaces.graph import Plot
class MutationHistogram(Plot):

    def __init__(self, session):
        self.mutation_set_name = ''

        Plot.__init__(self, session, tool_name = 'Deep mutational scan histogram')

        tw = self.tool_window
        tw.fill_context_menu = self._fill_context_menu
        self.canvas.mousePressEvent = tw._show_context_menu  # Show menu on left click.

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

        self.set_plot_data(score_name, mutation_set_name, bins = self._bins,
                           curve = self._smooth_curve, smooth_width = self._smooth_width, smooth_bins = self._smooth_bins,
                           scale = self._yscale)

        if scale == 'log':
            self._set_log_scale()

    def set_plot_data(self, score_name, mutation_set_name, bins = 20,
                      curve = True, smooth_bins = 20, smooth_width = None, scale = 'linear'):
        from .ms_data import mutation_scores
        scores = mutation_scores(self.session, mutation_set_name)
        score_values = scores.score_values(score_name)

        self._score_menu.value = score_name
        self._mutation_set_menu.value = self.mutation_set_name = scores.name

        values = [value for res_num, from_aa, to_aa, value in score_values.all_values()]

        self._set_values(values, title=scores.name, x_label=score_name, bins=bins,
                         smooth_curve=curve, smooth_width=smooth_width, smooth_bins=smooth_bins, yscale=scale)
        return values
    
    def _set_values(self, scores, title = '', x_label = '', bins = 20,
                   smooth_curve = False, smooth_width = None, smooth_bins = 200, yscale = 'linear'):
        a = self.axes
        a.clear()
        a.hist(scores, bins=bins)
        a.set_title(title)
        a.set_xlabel(x_label)
        a.set_ylabel('Count')
        if smooth_curve:
            x, y = gaussian_histogram(scores, sdev = smooth_width, bins = smooth_bins)
            y *= (smooth_bins / bins)	# Scale to match histogram bar height
            a.plot(x, y)
        # Remember state for session saving
        self._scores, self._bins, self._smooth_curve, self._smooth_width, self._smooth_bins = \
            scores, bins, smooth_curve, smooth_width, smooth_bins
        self.canvas.draw()
        ymin, self._ymax = a.get_ylim()
        if yscale == 'log':
            self._set_log_scale()

    def tight_layout(self):
        # Don't hide axes and reduce padding
        pass

    def _fill_context_menu(self, menu, x, y):
        if self._yscale == 'linear':
            self.add_menu_entry(menu, 'Log scale', self._set_log_scale)
        else:
            self.add_menu_entry(menu, 'Linear scale', self._set_linear_scale)

        self.add_menu_separator(menu)
        self.add_menu_entry(menu, 'New plot', self._copy_plot)
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

    def _copy_plot(self):
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
                'version': '1'}
        return data

    @classmethod
    def restore_snapshot(cls, session, data):
        hp = cls(session)
        hp.mutation_set_name = hp._mutation_set_menu.value = data['mutation_set_name']
        hp._score_menu.value = data.get('score_name', '')
        hp._set_values(data['scores'], title = data['title'], x_label = data['x_label'], bins = data['bins'],
                       smooth_curve = data['smooth_curve'], smooth_width = data['smooth_width'],
                       smooth_bins = data['smooth_bins'], yscale = data.get('yscale', 'linear'))
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
                   ('curve', BoolArg),
                   ('smooth_width', FloatArg),
                   ('smooth_bins', IntArg),
                   ('scale', EnumOf(['linear', 'log'])),
                   ('replace', BoolArg),
                   ],
        synopsis = 'Show histogram of mutation scores'
    )
    register('mutationscores histogram', desc, mutation_scores_histogram, logger=logger)
