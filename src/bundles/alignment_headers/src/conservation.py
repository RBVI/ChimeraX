# vim: set expandtab ts=4 sw=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

"""Find conservation sequences"""

from .header_sequence import DynamicHeaderSequence
from chimerax.seqalign import clustal_strong_groups, clustal_weak_groups

class Conservation(DynamicHeaderSequence):
    name = "Conservation"
    ident = "conservation"
    sort_val = 1.7

    STYLE_PERCENT = "identity histogram"
    STYLE_CLUSTAL_CHARS = "Clustal characters"
    STYLE_AL2CO = "AL2CO"
    styles = (STYLE_PERCENT, STYLE_CLUSTAL_CHARS, STYLE_AL2CO)

    AL2CO_cite = ["Pei, J. and Grishin, N.V. (2001)",
            "AL2CO: calculation of positional conservation in a protein sequence alignment",
            "Bioinformatics, 17, 700-712."]
    AL2CO_cite_prefix="Publications using AL2CO conservation measures should cite:"
    save_file_preamble = '\n# '.join(['# ' + AL2CO_cite_prefix] + AL2CO_cite)

    def __init__(self, alignment, *args, **kw):
        # need access to settings early, so replicate code in HeaderSequence
        self.alignment = alignment
        if not hasattr(self.__class__, 'settings'):
            self.__class__.settings = self.make_settings(alignment.session)
        self._set_update_vars(self.settings.style)
        self.handler_ID = self.settings.triggers.add_handler('setting changed', self._setting_changed_cb)
        super().__init__(alignment, *args, eval_while_hidden=True, **kw)

    def add_options(self, options_container, *, category=None, verbose_labels=True):
        from Qt.QtWidgets import QVBoxLayout
        from Qt.QtCore import Qt
        option_data = self.option_data()
        self._add_options(options_container, category, verbose_labels, option_data)
        if category is None:
            args = ()
        else:
            args = (category,)
        self.al2co_options_widget, al2co_options = options_container.add_option_group(*args,
            group_label="AL2CO parameters", group_alignment=Qt.AlignLeft)
        from chimerax.sim_matrices import matrices, matrix_name_key_func
        matrix_names = list(matrices(self.alignment.session).keys())
        matrix_names.append("identity")
        matrix_names.sort(key=matrix_name_key_func)
        class Al2coMatrixOption(EnumOption):
            values = matrix_names
        from chimerax.ui.options import IntOption, FloatOption
        al2co_option_data = [
            ("frequency estimation method", 'al2co_freq', Al2coFrequencyOption, {},
                "Method to estimate position-specific amino acid frequencies"),
            ("conservation measure", 'al2co_cons', Al2coConservationOption, {},
                "Conservation calculation strategy"),
            ("averaging window", 'al2co_window', IntOption, {'min': 1},
                "Window size for conservation averaging"),
            ("gap fraction", 'al2co_gap', FloatOption, {'min': 0.0, 'max': 1.0},
                "Conservations are computed for columns only if the fraction of gaps is less than this value"),
        ]
        self._add_options(al2co_options, None, False, al2co_option_data)
        layout = QVBoxLayout()
        layout.addWidget(al2co_options, alignment=Qt.AlignLeft)
        from chimerax.ui.widgets import Citation
        layout.addWidget(Citation(self.alignment.session, '\n'.join(self.AL2CO_cite),
            prefix=self.AL2CO_cite_prefix, pubmed_id=11524371), alignment=Qt.AlignLeft)
        self.al2co_options_widget.setLayout(layout)
        self.al2co_sop_options_widget, al2co_sop_options = al2co_options.add_option_group(
            group_label="Sum-of-pairs parameters")
        al2co_sop_option_data = [
            ("matrix", 'al2co_matrix', Al2coMatrixOption, {},
                "Similarity matrix used by sum-of-pairs measure"),
            ("matrix transformation", 'al2co_transform', Al2coTransformOption, {},
                "Transform applied to similarity matrix as follows:\n"
                "\t%s: identity substitutions have same value\n"
                "\t%s: adjustment so that 2-sequence alignment yields\n"
                "\t\tsame score as in original matrix" % tuple(Al2coTransformOption.labels[1:]))
        ]
        self._add_options(al2co_sop_options, None, False, al2co_sop_option_data)
        sop_layout = QVBoxLayout()
        sop_layout.addWidget(al2co_sop_options)
        self.al2co_sop_options_widget.setLayout(sop_layout)

        if self.settings.style == self.STYLE_AL2CO:
            if self.settings.al2co_cons != 2:
                self.al2co_sop_options_widget.hide()
        else:
            self.al2co_options_widget.hide()

    def destroy(self):
        self.handler_ID.remove()
        super().destroy()

    def evaluate(self, pos):
        # this will never get called if style is STYLE_AL2CO
        if self.style == self.STYLE_PERCENT:
            if len(self.alignment.seqs) == 1:
                return 1.0
            return self.percent_identity(pos)
        values = [' ', '.', ':', '*']
        return values[self.clustal_type(pos)]

    def get_state(self):
        state = {
            'base state': super().get_state(),
            'style': self.style,
            'al2co params': {
                'freq': self.settings.al2co_freq,
                'cons': self.settings.al2co_cons,
                'window': self.settings.al2co_window,
                'gap': self.settings.al2co_gap,
                'matrix': self.settings.al2co_matrix,
                'transform': self.settings.al2co_transform,
            }
        }
        return state

    def num_options(self):
        return 1

    def option_data(self):
        return super().option_data() + [ ("style", 'style', ConservationStyleOption, {}, None) ]

    def position_color(self, pos):
        return 'black' if self.style == self.STYLE_CLUSTAL_CHARS else 'dark gray'

    def percent_identity(self, pos, for_histogram=False):
        """actually returns a fraction"""
        occur = {}
        for i in range(len(self.alignment.seqs)):
            let = self.alignment.seqs[i][pos]
            try:
                occur[let] += 1
            except KeyError:
                occur[let] = 1
        best = 0
        for let, num in occur.items():
            if not let.isalpha():
                continue
            if num > best:
                best = num
        if best == 0:
            return 0.0
        if for_histogram:
            return (best - 1) / (len(self.alignment.seqs) - 1)
        return best / len(self.alignment.seqs)

    def reevaluate(self, pos1=0, pos2=None, *, evaluation_func=None):
        if self.style == self.STYLE_AL2CO:
            self.depiction_val = self.hist_infinity
        elif self.style == self.STYLE_PERCENT:
            self.depiction_val = self._hist_percent
        else:
            if hasattr(self, 'depiction_val'):
                delattr(self, 'depiction_val')
        evaluation_func = self._reeval_al2co if self.style == self.STYLE_AL2CO else evaluation_func
        return super().reevaluate(pos1, pos2, evaluation_func=evaluation_func)

    @property
    def residue_attr_name(self):
        if self.style == self.STYLE_CLUSTAL_CHARS:
            return self.ATTR_PREFIX + "clustal_" + self.ident
        return self.ATTR_PREFIX + self.ident

    def set_state(self, state):
        super().set_state(state['base state'])
        for param, val in state['al2co params'].items():
            setattr(self.settings, 'al2co_' + param, val)
        self.style = state['style']

    def settings_info(self):
        name, defaults = super().settings_info()
        from chimerax.core.commands import EnumOf, FloatArg, IntArg, Bounded, PositiveIntArg, BoolArg
        from chimerax.sim_matrices import matrices
        matrix_names = list(matrices(self.alignment.session).keys())
        defaults.update({
            'style': (EnumOf(self.styles), self.STYLE_AL2CO),
            'al2co_freq': (Bounded(IntArg, min=0, max=2), 2),
            'al2co_cons': (Bounded(IntArg, min=0, max=2), 0),
            'al2co_window': (PositiveIntArg, 1),
            'al2co_gap': (Bounded(FloatArg, min=0, max=1), 0.5),
            'al2co_matrix': (EnumOf(matrix_names), "BLOSUM-62"),
            'al2co_transform': (Bounded(IntArg, min=0, max=2), 0),
            'initially_shown': (BoolArg, True),
        })
        return "conservation sequence header", defaults

    @property
    def style(self):
        return self.settings.style

    @style.setter
    def style(self, style):
        if self.settings.style == style:
            return
        self._set_update_vars(style)
        self.settings.style = style

    @property
    def value_type(self):
        if self.style == self.STYLE_CLUSTAL_CHARS:
            return str
        return float

    def clustal_type(self, pos):
        conserve = None
        for i in range(len(self.alignment.seqs)):
            char = self.alignment.seqs[i][pos].upper()
            if conserve is None:
                conserve = char
                continue
            if char != conserve:
                break
        else:
            return 3

        for group in clustal_strong_groups:
            for i in range(len(self.alignment.seqs)):
                char = self.alignment.seqs[i][pos].upper()
                if char not in group:
                    break
            else:
                return 2

        for group in clustal_weak_groups:
            for i in range(len(self.alignment.seqs)):
                char = self.alignment.seqs[i][pos].upper()
                if char not in group:
                    break
            else:
                return 1

        return 0

    def _hist_percent(self, pos):
        return self.percent_identity(pos, for_histogram=True)

    def _reeval_al2co(self, pos1, pos2):
        if len(self.alignment.seqs) == 1:
            self[:] = [100.0] * len(self.alignment.seqs[0])
            return
        session = self.alignment.session
        self[:] = []
        # sequence names in the alignment may contain characters that cause AL2CO to barf,
        # so make a temporary alignment with sanitized names.
        # Also, AL2CO doesn't like spaces in the alignment (which occur in HSSP files), replace with gap
        from copy import copy
        sane_seqs = [copy(seq) for seq in self.alignment.seqs]
        for i, sseq in enumerate(sane_seqs):
            sseq.name = str(i)
            sseq.characters = sseq.characters.replace(' ', '.')

        temp_alignment = session.alignments.new_alignment(sane_seqs, False, auto_associate=False, name="temp", create_headers=False)
        from tempfile import NamedTemporaryFile
        temp_stream = NamedTemporaryFile(mode='w', encoding='utf8', suffix=".aln", delete=False)
        temp_alignment.save(temp_stream, format_name="aln")
        file_name = temp_stream.name
        temp_stream.close()
        session.alignments.destroy_alignment(temp_alignment)
        import os.path
        command = [os.path.join(os.path.dirname(__file__), "bin", "al2co.exe"),
            "-i", file_name,
            "-f", str(self.settings.al2co_freq),
            "-c", str(self.settings.al2co_cons),
            "-w", str(self.settings.al2co_window),
            "-g", str(self.settings.al2co_gap) ]
        if self.settings.al2co_cons == 2:
            command += ["-m", str(self.settings.al2co_transform)]
            from chimerax.sim_matrices import matrix_files
            matrix_lookup = matrix_files(session.logger)
            if self.settings.al2co_matrix in matrix_lookup:
                command += [ "-s", matrix_lookup[self.settings.al2co_matrix] ]
        try:
            import subprocess
            result = subprocess.run(command, capture_output=True, text=True, check=True)
        finally:
            import os
            os.unlink(file_name)
        for line in result.stdout.splitlines():
            if len(self) == len(self.alignment.seqs[0]):
                break
            line = line.strip()
            if line.endswith("zero"):
                # variance is zero
                continue
            if line.endswith("position"):
                # one or fewer columns have values
                self[:] = [0.0] * len(self.alignment.seqs[0])
                delattr(self, 'depiction_val')
                break
            if line[-1] == "*":
                self.append(None)
                continue
            self.append(float(line.split()[-1]))
        if len(self) != len(self.alignment.seqs[0]):
            # failure, possibly due to no variance in alignment
            self[:] = [1.0] * len(self.alignment.seqs[0])

    def _set_update_vars(self, style):
        self.single_column_updateable, self.fast_update = (False, False) \
            if style == self.STYLE_AL2CO else (True, True)

    def _setting_changed_cb(self, trig_name, trig_data):
        attr_name, prev_val, new_val = trig_data
        if hasattr(self, 'al2co_options_widget'):
            if attr_name == "style":
                self.al2co_options_widget.setHidden(new_val != self.STYLE_AL2CO)
            elif attr_name == "al2co_cons":
                self.al2co_sop_options_widget.setHidden(new_val != 2)
        self.reevaluate()

from chimerax.ui.options import EnumOption, SymbolicEnumOption
class ConservationStyleOption(EnumOption):
    values = Conservation.styles

class Al2coFrequencyOption(SymbolicEnumOption):
    labels = ["unweighted", "modified Henikoff & Henikoff", "independent counts"]
    values = list(range(len(labels)))

class Al2coConservationOption(SymbolicEnumOption):
    labels = ["entropy-based", "variance-based", "sum of pairs"]
    values = list(range(len(labels)))

class Al2coTransformOption(SymbolicEnumOption):
    labels = ["none", "normalization", "adjustment"]
    values = list(range(len(labels)))
