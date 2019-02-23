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
    sort_val = 1.7

    STYLE_PERCENT = "identity histogram"
    STYLE_CLUSTAL_CHARS = "Clustal characters"
    STYLE_AL2CO = "AL2CO"
    styles = (STYLE_PERCENT, STYLE_CLUSTAL_CHARS, STYLE_AL2CO)

    def __init__(self, alignment, *args, **kw):
        self.settings = get_settings(alignment.session)
        self._set_update_vars(self.settings.style)
        self.handler_ID = self.settings.triggers.add_handler('setting changed', self._setting_changed_cb)
        self.al2co_options_widget = None
        super().__init__(alignment, *args, eval_while_hidden=True, **kw)

    def add_options(self, options_container, *, category=None, verbose_labels=True):
        option_data =[ ("style", 'style', ConservationStyleOption, {}, None) ]
        self._add_options(options_container, category, verbose_labels, option_data)
        if category is None:
            args = ()
        else:
            args = (category,)
        self.al2co_options_widget, al2co_options = options_container.add_option_group(*args,
            group_label="AL2CO parameters")
        from chimerax.seqalign.sim_matrices import matrices
        matrix_names = list(matrices(self.alignment.session).keys())
        matrix_names.append("identity")
        matrix_names.sort(key=lambda x: x.lower())
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
            ("sum-of-pairs matrix", 'al2co_matrix', Al2coMatrixOption, {},
                "Similarity matrix used by sum-of-pairs measure"),
            ("matrix transformation", 'al2co_transform', Al2coTransformOption, {},
                "Transform applied to similarity matrix as follows:\n"
                "\t%s: identity substitutions have same value\n"
                "\t%s: adjustment so that 2-sequence alignment yields\n"
                "\t\tsame score as in original matrix" % tuple(Al2coTransformOption.labels[1:]))
        ]
        self._add_options(al2co_options, None, False, al2co_option_data)
        from PyQt5.QtWidgets import QVBoxLayout
        layout = QVBoxLayout()
        layout.addWidget(al2co_options)
        self.al2co_options_widget.setLayout(layout)
        if self.settings.style != self.STYLE_AL2CO:
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

    def num_options(self):
        return 1

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
    def style(self):
        return self.settings.style

    @style.setter
    def style(self, style):
        if self.settings.style == style:
            return
        self._set_update_vars(style)
        self.settings.style = style

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
        self[:] = [100.0] * len(self.alignment.seqs[0])
        return
        if len(self.alignment.seqs) == 1:
            self[:] = [100.0] * len(self.alignment.seqs[0])
            return
        self[:] = []
        from tempfile import NamedTemporaryFile
        temp_stream = NamedTemporaryFile(mode='w', encoding='utf8', suffix=".aln", delete=False)
        self.alignment.save(temp_stream, format_name="aln")
        file_name = temp_stream.name
        temp_stream.close()
        try:
            import subprocess
            import os.path
            #TODO
            """
            result = subprocess.run([os.path.join(os.path.dirname(__file__), "bin", "al2co.exe"),
                "-i", file_name,
                "-f", str(self.mav.prefs[AL2CO_FREQ]),
                "-c", str(self.mav.prefs[AL2CO_CONS]),
                "-w", str(self.mav.prefs[AL2CO_WINDOW]),
                "-g", str(self.mav.prefs[AL2CO_GAP]) ]
            """
        finally:
            import os
            os.unlink(file_name)
        """TODO
        import os, os.path
        chimeraRoot = os.environ.get("CHIMERA")
        command =  [ os.path.join(chimeraRoot, 'bin', 'al2co'),
                "-i", tfName,
                "-f", str(self.mav.prefs[AL2CO_FREQ]),
                "-c", str(self.mav.prefs[AL2CO_CONS]),
                "-w", str(self.mav.prefs[AL2CO_WINDOW]),
                "-g", str(self.mav.prefs[AL2CO_GAP]) ]
        if self.mav.prefs[AL2CO_CONS] == 2:
            command += ["-m", str(self.mav.prefs[AL2CO_TRANSFORM])]
            matrix = self.mav.prefs[AL2CO_MATRIX]
            from SmithWaterman import matrixFiles
            if matrix in matrixFiles:
                command += [ "-s", matrixFiles[matrix] ]
        from subprocess import Popen, PIPE, STDOUT
        alOut = Popen(command, stdin=PIPE, stdout=PIPE, stderr=STDOUT).stdout
        for line in alOut:
            if len(self) == len(self.mav.seqs[0]):
                break
            line = line.strip()
            if line.endswith("zero"):
                # variance is zero
                continue
            if line.endswith("position"):
                # one or fewer columns have values
                self[:] = [0.0] * len(self.mav.seqs[0])
                delattr(self, 'depictionVal')
                break
            if line[-1] == "*":
                self.append(None)
                continue
            self.append(float(line.split()[-1]))
        os.unlink(tfName)
        if len(self) != len(self.mav.seqs[0]):
            # failure, possibly due to no variance in alignment
            self[:] = [1.0] * len(self.mav.seqs[0])
        """

    def _set_update_vars(self, style):
        self.single_column_updateable, self.fast_update = (False, False) \
            if style == self.STYLE_AL2CO else (True, True)

    def _setting_changed_cb(self, trig_name, trig_data):
        attr_name, prev_val, new_val = trig_data
        if attr_name == "style":
            self.al2co_options_widget.setHidden(new_val != self.STYLE_AL2CO)
        self.reevaluate()

from chimerax.core.settings import Settings
class ConservationSettings(Settings):
    EXPLICIT_SAVE = {
        'style': Conservation.STYLE_AL2CO,
        'al2co_freq': 2,
        'al2co_cons': 0,
        'al2co_window': 1,
        'al2co_gap': 0.5,
        'al2co_matrix': "BLOSUM-62",
        'al2co_transform': 0,
    }

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

_settings = None
def get_settings(session):
    global _settings
    if _settings is None:
        _settings = ConservationSettings(session, "conservation alignment header")
    return _settings
