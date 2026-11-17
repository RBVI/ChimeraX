import pytest


@pytest.fixture
def pdbinfo(ensure_chimerax_initialized, monkeypatch):
    from chimerax.blastprotein.data_model import pdbinfo

    batches = []

    def fake_fetch(entry_ids):
        assert len(entry_ids) <= pdbinfo.MAX_ENTRIES_PER_QUERY
        batches.append(list(entry_ids))
        return [{"rcsb_id": e, "polymer_entities": []} for e in entry_ids]

    monkeypatch.setattr(pdbinfo, "_fetch_entries", fake_fetch)
    pdbinfo.batches = batches
    return pdbinfo


def test_entries_queried_once_per_entry(pdbinfo):
    info = pdbinfo.fetch_pdb_info(["1ABC_A", "1ABC_B", "2XYZ_A"])
    assert pdbinfo.batches == [["1ABC", "2XYZ"]]
    assert set(info) == {"1ABC_A", "1ABC_B", "2XYZ_A"}


def test_large_hit_lists_are_split_across_queries(pdbinfo):
    # RCSB rejects more than 1000 entry ids in one query
    hits = ["%04d_A" % i for i in range(2500)]
    info = pdbinfo.fetch_pdb_info(hits)
    assert [len(b) for b in pdbinfo.batches] == [1000, 1000, 500]
    assert len(info) == 2500
