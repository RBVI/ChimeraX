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

"""Show consensus sequence"""

from .header_sequence import DynamicHeaderSequence

class Consensus(DynamicHeaderSequence):
    name = "Consensus"
    sort_val = 1.3
    def __init__(self, alignment, refresh_callback, *, ignore_gaps=False, capitalize_at=0.8):
        self.capitalize_at = capitalize_at
        self.conserved = [False] * len(alignment.seqs[0])
        self._ignore_gaps = ignore_gaps
        super().__init__(alignment, refresh_callback)

    def evaluate(self, pos):
        occur = {}
        for i in range(len(self.alignment.seqs)):
            let = self.alignment.seqs[i][pos]
            if self._ignore_gaps and not let.isalpha():
                continue
            try:
                occur[let] += 1
            except KeyError:
                occur[let] = 1
        best = (0, None)
        for let, num in occur.items():
            if num > best[0]:
                best = (num, let)
            elif num == best[0] and not let.isalpha():
                # "gappy" characters win ties
                best = (num, let)
        num, let = best
        self.conserved[pos] = False
        if let is None:
            return ' '
        if num / len(self.alignment.seqs) >= self.capitalize_at:
            retlet = let.upper()
            if num == len(self.alignment.seqs):
                self.conserved[pos] = True
        else:
            retlet = let.lower()
        return retlet

    @property
    def ignore_gaps(self):
        return self._ignore_gaps

    @ignore_gaps.setter
    def ignore_gaps(self, ignore_gaps):
        if ignore_gaps != self._ignore_gaps:
            self._ignore_gaps = ignore_gaps
            if self.visible or self.evaluate_while_hidden:
                self.reevaluate()
            else:
                self._update_needed = True

    def position_color(self, pos):
        if self[pos].isupper():
            if self.conserved[pos]:
                return 'red'
            return 'purple'
        return 'black'

    def reevaluate(self):
        """sequences changed, possibly including length"""
        self.conserved = [False] * len(self.alignment.seqs[0])
        super().reevaluate()
