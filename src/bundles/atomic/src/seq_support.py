# vim: set expandtab shiftwidth=4 softtabstop=4:

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

from chimerax.core.toolshed import ProviderManager
from chimerax.core.state import State
import re

class SeqFeatureManager(ProviderManager):

    def __init__(self):
        self._data_source_info = {}
        super().__init__("sequence features")

    def add_provider(self, bundle_info, name):
        self._data_source_info[name] = bundle_info

    @property
    def data_sources(self):
        return self._data_source_info.keys()

    def get_features(self, seq, data_source):
        if data_source not in self._data_source_info:
            raise ValueError("Unknown sequence-feature data source: %s" % data_source)

class GenericSeqFeature(State):
    EVIDENCE_MARKER = 'evidence='
    def __init__(self, *args):
        if args:
            info, positions = args
            self.details = []
            self.evidence_codes = set()
            for line in info:
                if line.startswith(self.EVIDENCE_MARKER):
                    self.details.append(self._process_evidence(line))
                else:
                    self.details.append(line)
            self.positions = positions

    @classmethod
    def restore_snapshot(cls, session, data):
        feat = cls()
        feat.set_state_from_snapshot(data)
        return feat

    def set_state_from_snapshot(self, state):
        for attr_name, val in state.items():
            setattr(self, attr_name, val)

    def take_snapshot(self, session, flags):
        return {
            'details': self.details,
            'evidence_codes': self.evidence_codes,
            'positions': self.positions
        }

    def _process_evidence(self, line):
        try:
            codes = [int(x) for x in line[len(self.EVIDENCE_MARKER):].split()]
        except ValueError:
            return line
        self.evidence_codes.update(codes)
        new_line = "Evidence code%s: " % ("s" if len(codes) > 1 else "")
        new_line += " ".join(['<a href="https://evidenceontology.org/browse/#ECO_%07d">%d</a>'
            % (code, code) for code in codes])
        return new_line

class SeqVariantFeature(GenericSeqFeature):
    dbsnp_matcher = re.compile(r"\bdbSNP:rs(\d+)", re.ASCII)

    def __init__(self, *args):
        if args:
            info, positions = args
            processed_info = []
            self.dbsnp_ref_id = None
            for line in info:
                match = re.search(self.dbsnp_matcher, line)
                if match:
                    processed = line[:match.start()]
                    number = int(match.group(1))
                    self.dbsnp_ref_id = number
                    processed += '<a href="https://www.ncbi.nlm.nih.gov/snp/?term=rs%d">%s</a>' % (number,
                        match.group(0))
                    processed += line[match.end():]
                    processed_info.append(processed)
                else:
                    processed_info.append(line)
            super().__init__(processed_info, positions)

    def set_state_from_snapshot(self, state):
        super().set_state_from_snapshot(state.pop('base'))
        for attr_name, val in state.items():
            setattr(self, attr_name, val)

    def take_snapshot(self, session, flags):
        return {
            'base': super().take_snapshot(session, flags),
            'dbsnp_ref_id': self.dbsnp_ref_id
        }

def feature_type_to_class(ftype):
    return { 'sequence variant': SeqVariantFeature }.get(ftype, GenericSeqFeature)

_manager = None
def get_manager():
    global _manager
    if _manager is None:
        _manager = SeqFeatureManager()
    return _manager
