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

"""Header sequence to show RMSD of associated structures"""

carbon_alpha = "C\N{GREEK SMALL LETTER ALPHA}"

from .header_sequence import DynamicStructureHeaderSequence

class RMSD(DynamicStructureHeaderSequence):

    settings_name = "RMSD"
    ident = "rmsd"
    min_structure_relevance = 2

    def __init__(self, alignment, *args, **kw):
        from math import log
        self.scaling = log(0.5) / (-3.0)
        super().__init__(alignment, *args, **kw)
        from chimerax.atomic import get_triggers
        self.handlers = [
            self.settings.triggers.add_handler('setting changed', self._setting_changed_cb),
            get_triggers().add_handler('changes', self._atomic_changes_cb)
        ]
        self._set_name()

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
            'atoms': (EnumOf(RmsdDomainOption.values), carbon_alpha),
        })
        return "RMSD sequence header", defaults

    def reevaluate(self, pos1=0, pos2=None):
        if not self._shown and not self.eval_while_hidden:
            self._update_needed = True
            return
        if pos1 == 0 and pos2 == None:
            original_eval_chains = getattr(self, '_eval_chains', [])
            by_struct = {}
            for chain in self.alignment.associations:
                by_struct.setdefault(chain.structure, []).append(chain)
            chain_lists = list(by_struct.values())
            if len(chain_lists) < 2:
                self._eval_chains = [cl[0] for cl in chain_lists]
            else:
                with self.alignment_notifications_suppressed():
                    chain_lists.sort(key=lambda x: len(x))
                    cl1, cl2 = chain_lists[:2]
                    lowest = None
                    for c1 in cl1:
                        for c2 in cl2:
                            self._eval_chains = [c1, c2]
                            super().reevaluate()
                            vals = [v for v in self[:] if v is not None]
                            if not vals:
                                continue
                            avg = sum(vals) / len(vals)
                            if lowest is None or avg < lowest:
                                lowest = avg
                                best_chains = [c1, c2]
                    if lowest is None:
                        best_chains = [cl1[0], cl2[0]]
                    for cl in chain_lists[2:]:
                        lowest = None
                        for c in cl:
                            self._eval_chains = best_chains + [c]
                            super().reevaluate()
                            vals = [v for v in self[:] if v is not None]
                            if not vals:
                                continue
                            avg = sum(vals) / len(vals)
                            if lowest is None or avg < lowest:
                                lowest = avg
                                best_chain = c
                        if lowest is None:
                            best_chains.append(cl[0])
                        else:
                            best_chains.append(best_chain)
                    self._eval_chains = best_chains
                    if set(best_chains) != set(original_eval_chains):
                        self.alignment.session.logger.info("Chains used in RMSD evaluation for alignment"
                            " %s: %s" % (self.alignment, ', '.join(str(c) for c in sorted(best_chains))))
                # to force the refresh callback to happen...
                self.clear()
        super().reevaluate(pos1, pos2)

    def _gather_coords(self, pos):
        if self.atoms == carbon_alpha:
            bb_names = ["CA"]
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

    def _atomic_changes_cb(self, trig_name, changes):
        if 'scene_coord changed' not in changes.structure_reasons():
            return
        for chain in self.alignment.associations:
            if chain.structure in changes.modified_structures():
                self.reevaluate()
                break

    def _set_name(self):
        if self.atoms == carbon_alpha:
            self.name = carbon_alpha + " RMSD"
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
    values = [carbon_alpha, "backbone"]
