# Plot a histogram of mutation scores.
def mutation_scores_histogram(session, score_name, scores_name = None, subtract_fit = None,
                              bins = 20, curve = True, smooth_width = None,
                              type = 'all_mutations', above = None, below = None, replace = True):
    from .ms_data import mutation_scores
    scores = mutation_scores(session, scores_name)
    score_values = scores.score_values(score_name, subtract_fit = subtract_fit)

    res_nums = []
    res_scores = []
    score_names = []
    if type == 'all_mutations':
        for res_num, from_aa, to_aa, value in score_values.all_values():
            res_nums.append(res_num)
            res_scores.append(value)
            score_names.append(f'{from_aa}{res_num}{to_aa}')
    else:
        for res_num in score_values.residue_numbers():
            value = score_values.residue_value(res_num, value_type = type, above = above, below = below)
            if value is not None:
                res_nums.append(res_num)
                res_scores.append(value)
                score_names.append(f'{res_num}')
    
    from numpy import array, float32
    res_scores = array(res_scores, float32)

    chain = scores.chain
    if chain:
        resnum_to_res = {r.number:r for r in chain.existing_residues}
        residues = [resnum_to_res.get(res_num) for res_num in res_nums]
    else:
        residues = [None] * len(res_nums)

    if replace:
        plot = getattr(scores, '_last_mutation_scores_histogram', None)
        if plot and plot.tool_window.ui_area is None:
            plot is None
    if plot is None:
        plot = Histogram(session, title = 'Deep mutational scan histogram')
    scores._last_mutation_scores_histogram = plot

    plot.set_values(res_scores, residues, score_names=score_names,
                    title=scores.name, x_label=score_name, bins=bins,
                    smooth_curve=curve, smooth_width=smooth_width)

    in_chain = f' of chain {chain}' if chain else ''
    message = f'Plotted {len(res_scores)} scores{in_chain} for {score_name}'
    session.logger.info(message)

# TODO: Draw smooth curve by gaussian smoothing 1d bin array with map_filter code.
# TODO: Make mouse click select residues for histogram bar.
# TODO: Make mouse hover show mutation names for histogram bar in popup.
from chimerax.interfaces.graph import Plot
class Histogram(Plot):

    def __init__(self, session, title = 'Histogram'):
        Plot.__init__(self, session, tool_name = 'Deep Mutational Scan')
        self.tool_window.title = title
        self._highlight_color = (0,255,0,255)
        self._unhighlight_color = (150,150,150,255)

    def set_values(self, scores, residues, score_names = None, title = '', x_label = '', bins = 20,
                   smooth_curve = False, smooth_width = None, smooth_bins = 200):
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
        self.canvas.draw()

    def tight_layout(self):
        # Don't hide axes and reduce padding
        pass

    # TODO: Hook up context menu
    def fill_context_menu(self, menu, item):
        if item is not None:
            r = item.residue
            name = item.description
            self.add_menu_entry(menu, f'Select {name}', lambda self=self, r=r: self._select_residue(r))
            self.add_menu_entry(menu, f'Color {name}', lambda self=self, r=r: self._highlight_residue(r))
            self.add_menu_entry(menu, f'Zoom to {name}', lambda self=self, r=r: self._zoom_to_residue(r))
        else:
            self.add_menu_entry(menu, 'Save Plot As...', self.save_plot_as)

    def _select_residue(self, r):
        self._run_residue_command(r, 'select %s')
    def _highlight_residue(self, r):
        self._run_residue_command(r, 'color %s lime')
    def _zoom_to_residue(self, r):
        self._run_residue_command(r, 'view %s')
    def _run_residue_command(self, r, command):
        self._run_command(command % r.string(style = 'command'))
    def _run_command(self, command):
        from chimerax.core.commands import run
        run(self.session, command)

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

def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, StringArg, EnumOf, BoolArg, FloatArg, IntArg
    from .ms_data import ColumnValues
    desc = CmdDesc(
        required = [('score_name', StringArg)],
        keyword = [('scores_name', StringArg),
                   ('subtract_fit', StringArg),
                   ('bins', IntArg),
                   ('curve', BoolArg),
                   ('smooth_width', FloatArg),
                   ('type', EnumOf(('all_mutations',) + ColumnValues.residue_value_types)),
                   ('above', FloatArg),
                   ('below', FloatArg),
                   ('replace', BoolArg),
                   ],
        synopsis = 'Show histogram of mutation scores'
    )
    register('mutationscores histogram', desc, mutation_scores_histogram, logger=logger)
