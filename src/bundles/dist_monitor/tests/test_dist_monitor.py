expected_file_contents = """Model 1 is 3fx2

Distance information:
/A HOH 300 O <-> HOH 396 O:  3.672
/A ASP 62 OD2 <-> TYR 98 OH:  5.736
/A ASP 62 OD1 <-> TYR 100 OH:  8.906
"""

def test_dist_monitor(test_production_session, tmp_path):
    from chimerax.core.commands import run

    session = test_production_session
    tmpfile = tmp_path / "output.txt"
    run(session, "open 3fx2")
    run(session, "distance /A:62@OD1 /A:100@OH")
    run(session, "distance /A:62@OD2 /A:98@OH")
    run(session, "distance /A:300@O /A:396@O")
    run(session, "distance save %s" % str(tmpfile.absolute()))
    with open(tmpfile, 'r') as f:
        assert f.read() == expected_file_contents
