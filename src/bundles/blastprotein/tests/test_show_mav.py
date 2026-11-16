from types import SimpleNamespace

import pytest


class _Alignments:
    def __init__(self):
        self.calls = []

    def new_alignment(self, seqs, name):
        self.calls.append([(s.name, s.characters) for s in seqs])


@pytest.fixture
def results(ensure_chimerax_initialized):
    sequences = {
        0: ("query", SimpleNamespace(sequence="MKV-")),
        3: ("hitA", SimpleNamespace(sequence="MRV-")),
        9: ("hitB", SimpleNamespace(sequence="M-V-")),
    }
    return SimpleNamespace(
        _sequences=sequences,
        table=SimpleNamespace(data=[{"id": 3}, {"id": 9}]),
        _instance_name="bp1",
        _viewer_index=1,
        session=SimpleNamespace(alignments=_Alignments()),
    )


def _show_mav(results, selections):
    from chimerax.blastprotein.ui.results import BlastProteinResults

    BlastProteinResults._show_mav(results, selections)
    return results.session.alignments.calls[-1]


def test_show_mav_without_selection_aligns_every_match(results):
    assert _show_mav(results, []) == [
        ("query", "MKV"),
        ("hitA", "MRV"),
        ("hitB", "M-V"),
    ]


def test_show_mav_with_selection_aligns_only_selected(results):
    assert _show_mav(results, [{"id": 9}]) == [("query", "MKV"), ("hitB", "M-V")]
