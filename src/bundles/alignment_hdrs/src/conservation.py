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

    CSV_PERCENT = "identity histogram"
    CSV_CLUSTAL_CHARS = "Clustal characters"
    CSV_AL2CO = "AL2CO"
    styles = (CSV_PERCENT, CSV_CLUSTAL_CHARS)

    def __init__(self, *args, style=CSV_PERCENT, **kw):
        self._style = style
        self._set_update_vars(style)
        super().__init__(*args, **kw)

    def evaluate(self, pos):
        # this will never get called if style is CSV_AL2CO
        if self.style == self.CSV_PERCENT:
            return self.percent_identity(pos)
        values = [' ', '.', ':', '*']
        return values[self.clustal_type(pos)]

    def reevaluate(self):
        if self.style == self.CSV_AL2CO:
            self.depiction_val = self.hist_infinity
        elif self.style == self.CSV_PERCENT:
            self.depiction_val = self._hist_percent
        else:
            if hasattr(self, 'depiction_val'):
                delattr(self, 'depiction_val')
        if self.style != self.CSV_AL2CO:
            return DynamicHeaderSequence.reevaluate(self)
        if len(self.alignment.seqs) == 1:
            if self.style == self.CSV_AL2CO:
                self[:] = [100.0] * len(self.alignment.seqs[0])
            else:
                self[:] = [1.0] * len(self.alignment.seqs[0])
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

    def position_color(self, pos):
        return 'black' if self.style == self.CSV_CLUSTAL_CHARS else 'dark gray'

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

    @property
    def style(self):
        return self._style

    @style.setter
    def style(self, style):
        if self._style == style:
            return
        self._style = style
        self._set_update_vars(style)
        if self.visible or self.update_while_hidden:
            self.reevaluate()
        else:
            self._update_needed = True

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

    def _set_update_vars(self, style):
        self.single_column_updateable, self.fast_update = (False, False) \
            if style == self.CSV_AL2CO else (True, True)
