import json
from unittest.mock import patch

from chimerax.similarstructures.mmseqs2_search import fetch_pdb_entity_info


class _FakeResponse:
    def __init__(self, payload):
        self._data = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._data

    def close(self):
        pass


def test_fetch_pdb_entity_info_sends_json_graphql_query():
    # RCSB no longer accepts Content-Type: application/graphql (ticket #21099)
    captured = {}

    def fake_urlopen(req):
        captured["req"] = req
        return _FakeResponse({"data": {"polymer_entities": []}})

    with patch("urllib.request.urlopen", fake_urlopen):
        info = fetch_pdb_entity_info({"1ABC_1"})

    req = captured["req"]
    assert req.get_header("Content-type") == "application/json"
    body = json.loads(req.data.decode("utf-8"))
    assert '"1ABC_1"' in body["query"]
    assert "polymer_entities" in body["query"]
    assert info == {"data": {"polymer_entities": []}}
