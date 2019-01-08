expected_file_contents = """Model 1 is 3fx2

Distance information:
/A HOH 300 O <-> HOH 396 O:  3.672
/A ASP 62 OD2 <-> TYR 98 OH:  5.736
/A ASP 62 OD1 <-> TYR 100 OH:  8.906
"""
from tempfile import NamedTemporaryFile
tempf = NamedTemporaryFile(delete=False)
tempf.close()
from chimerax.core.commands import run, quote_if_necessary
run(session, "open 3fx2 ; distance /A:62@OD1 /A:100@OH ; distance /A:62@OD2 /A:98@OH ; distance /A:300@O /A:396@O; distance save %s" % quote_if_necessary(tempf.name))
f = open(tempf.name)
file_contents = f.read()
f.close()
import os
os.unlink(tempf.name)
if file_contents != expected_file_contents:
	raise SystemExit("Distance save file contents not same as expected")
