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

    def __init__(self, alignment, *args, **kw):
        from math import log
        self.scaling = log(0.5) / (-3.0)
        super().__init__(alignment, *args, eval_while_hidden=True, **kw)
        from chimerax.core.models import MODEL_POSITION_CHANGED
        self.handlers = [
            self.settings.triggers.add_handler('setting changed', self._setting_changed_cb),
            alignment.session.triggers.add_handler(MODEL_POSITION_CHANGED, self._model_position_changed_cb)
        ]
        self._set_name()

    def add_options(self, options_container, *, category=None, verbose_labels=True):
        self._add_options(options_container, category, verbose_labels, self.option_data())

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
        from chimerax.core.geometry import distance_squared
        for coords in self._gather_coords(pos):
            for i, crd1 in enumerate(coords):
                for crd2 in coords[i+1:]:
                    sum += distance_squared(crd1, crd2)
                n += (len(coords) * (len(coords)-1)) // 2
        if n == 0:
            return None
        from math import sqrt
        return sqrt(sum / n)

    def num_options(self):
        return 1

    def option_data(self):
        return super().option_data() + [ ("Atoms used for RMSD", 'atoms', RmsdDomainOption, {},
            "The atoms from each residue that are used in computing the RMSD") ]

    def position_color(self, pos):
        return 'dark gray'

    def settings_info(self):
        name, defaults = super().settings_info()
        defaults.update({
            'atoms': carbon_alpha,
        })
        return "RMSD sequence header", defaults

    def _gather_coords(self, pos):
        if self.atoms == carbon_alpha:
            bb_names = ["CA"]
        else:
            bb_names = None
        residues = []
        for chain, seq in self.alignment.associations.items():
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

    def _model_position_changed_cb(self, trig_name, model):
        for chain in self.alignment.associations:
            if chain.structure == model:
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
            self.refresh_callback(self, "name")

from chimerax.ui.options import EnumOption, SymbolicEnumOption
class RmsdDomainOption(EnumOption):
    values = [carbon_alpha, "backbone"]
