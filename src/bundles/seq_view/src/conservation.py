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
from .settings import ALIGNMENT_PREFIX, CSV_AL2CO, CSV_PERCENT, CSV_CLUSTAL_HIST, CSV_CLUSTAL_CHARS
#from prefs import CONSERVATION_STYLE, \
#        CSV_AL2CO, CSV_PERCENT, CSV_CLUSTAL_HIST, CSV_CLUSTAL_CHARS, \
#        AL2CO_FREQ, AL2CO_CONS, AL2CO_WINDOW, AL2CO_GAP, AL2CO_MATRIX, \
#        AL2CO_TRANSFORM

class Conservation(DynamicHeaderSequence):
    name = "Conservation"
    sort_val = 1.7
    def align_change(self, left, right):
        if getattr(self.sv.settings, ALIGNMENT_PREFIX + "conservation_style") == CSV_AL2CO:
            self.reevaluate()
        else:
            DynamicHeaderSequence.align_change(self, left, right)
    
    def evaluate(self, pos):
        if self.style == CSV_PERCENT:
            return self.percent_identity(pos)
        if self.style == CSV_CLUSTAL_HIST:
            values = [0.0, 0.33, 0.67, 1.0]
        else:
            values = [' ', '.', ':', '*']
        return values[self.clustal_type(pos)]

    def fast_update(self):
        return self.style != CSV_AL2CO

    def reevaluate(self):
        if self.style == CSV_AL2CO:
            self.depiction_val = self.hist_infinity
        elif self.style == CSV_PERCENT:
            self.depiction_val = self._hist_percent
        else:
            if hasattr(self, 'depiction_val'):
                delattr(self, 'depiction_val')
        if self.style != CSV_AL2CO:
            return DynamicHeaderSequence.reevaluate(self)
        if len(self.sv.alignment.seqs) == 1:
            if self.style == CSV_AL2CO:
                self[:] = [100.0] * len(self.sv.alignment.seqs[0])
            else:
                self[:] = [1.0] * len(self.sv.alignment.seqs[0])
            return
        """TODO
        self[:] = []
        from formatters.saveALN import save, extension
        from tempfile import mkstemp
        tfHandle, tfName = mkstemp(extension)
        import os
        os.close(tfHandle)
        import codecs
        tf = codecs.open(tfName, "w", "utf8")
        save(tf, None, self.mav.seqs, None)
        tf.close()
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
        return 'black' if self.style == CSV_CLUSTAL_CHARS else 'dark gray'

    def percent_identity(self, pos, for_histogram=False):
        """actually returns a fraction"""
        occur = {}
        for i in range(len(self.sv.alignment.seqs)):
            let = self.sv.alignment.seqs[i][pos]
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
            return (best - 1) / (len(self.sv.alignment.seqs) - 1)
        return best / len(self.sv.alignment.seqs)

    @property
    def style(self):
        return getattr(self.sv.settings, ALIGNMENT_PREFIX + "conservation_style")

    def clustal_type(self, pos):
        conserve = None
        for i in range(len(self.sv.alignment.seqs)):
            char = self.sv.alignment.seqs[i][pos].upper()
            if conserve is None:
                conserve = char
                continue
            if char != conserve:
                break
        else:
            return 3

        for group in clustal_strong_groups:
            for i in range(len(self.sv.alignment.seqs)):
                char = self.sv.alignment.seqs[i][pos].upper()
                if char not in group:
                    break
            else:
                return 2

        for group in clustal_weak_groups:
            for i in range(len(self.sv.alignment.seqs)):
                char = self.sv.alignment.seqs[i][pos].upper()
                if char not in group:
                    break
            else:
                return 1

        return 0

    def _hist_percent(self, pos):
        return self.percent_identity(pos, for_histogram=True)
