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
from .settings import CSN_MAJ_NOGAP, ALIGNMENT_PREFIX

class Consensus(DynamicHeaderSequence):
    name = "Consensus"
    sortVal = 1.3
    def __init__(self, sv, capitalize_at=0.8):
        self.capitalize_at = capitalize_at
        self.conserved = [False] * len(sv.alignment.seqs[0])
        super().__init__(sv)

    def evaluate(self, pos):
        occur = {}
        for i in range(len(self.sv.alignment.seqs)):
            let = self.sv.alignment.seqs[i][pos]
            if getattr(self.sv.settings, ALIGNMENT_PREFIX + "consensus_style") == CSN_MAJ_NOGAP \
            and not let.isalpha():
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
        if num / len(self.sv.alignment.seqs) >= self.capitalize_at:
            retlet = let.upper()
            if num == len(self.sv.alignment.seqs):
                self.conserved[pos] = True
        else:
            retlet = let.lower()
        return retlet

    def position_color(self, pos):
        if self[pos].isupper():
            if self.conserved[pos]:
                return 'red'
            return 'purple'
        return 'black'

    def reevaluate(self):
        """sequences changed, possibly including length"""
        self.conserved = [0] * len(self.sv.alignment.seqs[0])
        super().reevaluate()
