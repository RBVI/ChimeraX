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

"""Header sequence to show RMSD of associated structures"""

carbon_alpha = "C\N{GREEK SMALL LETTER ALPHA}"
principal_atom = carbon_alpha + "/C4'"

from .header_sequence import DynamicStructureHeaderSequence

class RMSD(DynamicStructureHeaderSequence):

    settings_name = "RMSD"
    ident = "rmsd"
    min_structure_relevance = 2

    def __init__(self, alignment, *args, **kw):
        from math import log
        self.scaling = log(0.5) / (-3.0)
        self._eval_chains = []
        super().__init__(alignment, *args, **kw)
        from chimerax.atomic import get_triggers
        self.handlers = [
            self.settings.triggers.add_handler('setting changed', self._setting_changed_cb),
        ]
        self._polymer_type = None
        self._set_name()

    def alignment_notification(self, note_name, note_data):
        if note_name == self.alignment.NOTE_RMSD_UPDATE:
            self.reevaluate()
        elif note_name == self.alignment.NOTE_REALIGNMENT:
            with self.alignment_notifications_suppressed():
                self.reevaluate()

    @property
    def atoms(self):
        return self.settings.atoms

    @atoms.setter
    def atoms(self, domain):
        if self.settings.atoms == domain:
            return
        self.settings.atoms = domain

    def destroy(self):
        for handler in self.handlers:
            handler.remove()
        super().destroy()

    def depiction_val(self, pos):
        val = self[pos]
        if val is None:
            return ' '
        from math import exp
        return 1.0 - exp(0.0 - self.scaling * val)

    def evaluate(self, pos):
        sum = 0.0
        n = 0
        from chimerax.geometry import distance_squared
        for coords in self._gather_coords(pos):
            for i, crd1 in enumerate(coords):
                for crd2 in coords[i+1:]:
                    sum += distance_squared(crd1, crd2)
            n += (len(coords) * (len(coords)-1)) // 2
        if n == 0:
            return None
        from math import sqrt
        return sqrt(sum / n)

    def get_state(self):
        state = {
            'base state': super().get_state(),
            'atoms': self.settings.atoms,
        }
        return state

    def num_options(self):
        return 1

    def option_data(self):
        return super().option_data() + [
            ("atoms used for RMSD", 'atoms', RmsdDomainOption, {},
                "The atoms from each residue that are used in computing the RMSD"),
        ]

    def position_color(self, pos):
        return 'dark gray'

    def set_state(self, state):
        super().set_state(state['base state'])
        self.settings.atoms = state['atoms']

    def settings_info(self):
        name, defaults = super().settings_info()
        from chimerax.core.commands import EnumOf
        defaults.update({
            'atoms': (EnumOf(RmsdDomainOption.values), principal_atom),
        })
        return "RMSD sequence header", defaults

    def reevaluate(self, pos1=0, pos2=None):
        if not self._shown and not self.eval_while_hidden:
            self._update_needed = True
            return
        if pos1 == 0 and pos2 == None:
            new_eval_chains = self.alignment.rmsd_chains
            if new_eval_chains and set(new_eval_chains) != set(self._eval_chains):
                self.alignment.session.logger.info("Chains used in RMSD evaluation for alignment"
                    " %s: %s" % (self.alignment, ', '.join(str(c) for c in sorted(new_eval_chains))))
            self._eval_chains = new_eval_chains
            # to force the refresh callback to happen...
            self.clear()
        super().reevaluate(pos1, pos2)

    def _gather_coords(self, pos):
        if self.atoms == principal_atom:
            bb_names = ["CA", "C4'"]
        else:
            bb_names = None
        residues = []
        for chain in self._eval_chains:
            seq = self.alignment.associations[chain]
            ungapped = seq.gapped_to_ungapped(pos)
            if ungapped is None:
                continue
            match_map = seq.match_maps[chain]
            try:
                r = match_map[ungapped]
            except KeyError:
                continue
            if r:
                residues.append(r)
                if not self._polymer_type:
                    if r.polymer_type != r.PT_NONE:
                        self._polymer_type = r.polymer_type
                        self._set_name()
                if not bb_names:
                    if r.polymer_type == r.PT_AMINO:
                        bb_names = r.aa_max_backbone_names
                    elif r.polymer_type == r.PT_NUCLEIC:
                        bb_names = r.na_max_backbone_names
        if not bb_names:
            return []

        coord_lists = []
        for bb_name in bb_names:
            coords = []
            for r in residues:
                a = r.find_atom(bb_name)
                if a:
                    coords.append(a.scene_coord)
            if len(coords) > 1:
                coord_lists.append(coords)
        return coord_lists

    def _set_name(self):
        if self.atoms == principal_atom:
            from chimerax.atomic import Residue
            if self._polymer_type == Residue.PT_NUCLEIC:
                prefix = "C4'"
            else:
                prefix = carbon_alpha
            self.name = prefix + " RMSD"
        else:
            self.name = "Backbone RMSD"

    def _setting_changed_cb(self, trig_name, trig_data):
        attr_name, prev_val, new_val = trig_data
        if attr_name == "atoms":
            self.reevaluate()
            self._set_name()
            self.notify_alignment(self.alignment.NOTE_HDR_NAME)

from chimerax.ui.options import EnumOption
class RmsdDomainOption(EnumOption):
    values = [principal_atom, "backbone"]
