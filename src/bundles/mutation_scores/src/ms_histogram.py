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
                              bins = 20, curve = True, smooth_width = None,
                              replace = True):
    from .ms_data import mutation_scores
    scores = mutation_scores(session, mutation_set)
    score_values = scores.score_values(score_name)

    values = [value for res_num, from_aa, to_aa, value in score_values.all_values()]

    plot = _find_mutation_histogram(session, scores.name) if replace else None
    if plot is None:
        plot = MutationHistogram(session, scores.name)

    plot.set_values(values, title=scores.name, x_label=score_name, bins=bins,
                    smooth_curve=curve, smooth_width=smooth_width)

    range = f'having range {"%.3g"%min(values)} to {"%.3g"%max(values)}' if len(values) > 0 else ''
    message = f'Plotted {len(values)} scores {range} for {score_name}'
    session.logger.info(message)

from chimerax.interfaces.graph import Plot
class MutationHistogram(Plot):

    def __init__(self, session, mutation_set_name):
        self.mutation_set_name = mutation_set_name
        Plot.__init__(self, session, tool_name = 'Deep mutational scan histogram')
        self._highlight_color = (0,255,0,255)
        self._unhighlight_color = (150,150,150,255)

    def set_values(self, scores, title = '', x_label = '', bins = 20,
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
        # Remember state for session saving
        self._scores, self._bins, self._smooth_curve, self._smooth_width, self._smooth_bins = \
            scores, bins, smooth_curve, smooth_width, smooth_bins
        self.canvas.draw()

    def tight_layout(self):
        # Don't hide axes and reduce padding
        pass

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
    
    # ---------------------------------------------------------------------------
    # Session save and restore.
    #
    SESSION_SAVE = True
    def take_snapshot(self, session, flags):
        axes = self.axes
        data = {'mutation_set_name': self.mutation_set_name,
                'scores': self._scores,
                'title': axes.get_title(),
                'x_label': axes.get_xlabel(),
                'bins': self._bins,
                'smooth_curve': self._smooth_curve,
                'smooth_width': self._smooth_width,
                'smooth_bins': self._smooth_bins,
                'version': '1'}
        return data

    @classmethod
    def restore_snapshot(cls, session, data):
        hp = cls(session, data['mutation_set_name'])
        hp.set_values(data['scores'], title = data['title'], x_label = data['x_label'], bins = data['bins'],
                      smooth_curve = data['smooth_curve'], smooth_width = data['smooth_width'],
                      smooth_bins = data['smooth_bins'])
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
             if isinstance(tool, MutationHistogram) and tool.mutation_set_name == mutation_set_name]
    return hists[-1] if hists else None

def register_command(logger):
    from chimerax.core.commands import CmdDesc, register, StringArg, BoolArg, FloatArg, IntArg
    desc = CmdDesc(
        required = [('score_name', StringArg)],
        keyword = [('mutation_set', StringArg),
                   ('bins', IntArg),
                   ('curve', BoolArg),
                   ('smooth_width', FloatArg),
                   ('replace', BoolArg),
                   ],
        synopsis = 'Show histogram of mutation scores'
    )
    register('mutationscores histogram', desc, mutation_scores_histogram, logger=logger)
