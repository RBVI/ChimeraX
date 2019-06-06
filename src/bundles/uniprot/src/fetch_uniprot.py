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

class InvalidAccessionError(ValueError):
	pass

def fetch_uniprot(session, ident, ignore_cache=False):
	'Fetch UniProt data'

	from chimerax.core.errors import UserError, CancelOperation
	try:
		accession = map_uniprot_ident(ident)
		seq_string, full_name, features = fetch_uniprot_accession_info(session, accession,
			ignore_cache=ignore_cache)
	except InvalidAccessionError as e:
		raise UserError(str(e))
	except CancelOperation:
		session.logger.status("Fetch of %s cancelled" % ident)
		return
	from chimerax.atomic import Sequence
	seq = Sequence(name=ident)
	seq.extend(seq_string)
	session.logger.status("Opening UniProt %s" % ident)
	session.alignments.new_alignment([seq], ident)
	return [], "Opened UniProt %s" % ident

def map_uniprot_ident(ident):
	from urllib.parse import urlencode
	params = {
		'from': 'ACC+ID',
		'to': 'ACC',
		'format': 'tab',
		'query': ident
	}
	data = urlencode(params)
	from urllib.request import Request, urlopen
	from urllib.error import HTTPError
	request = Request("https://www.uniprot.org/uploadlists/", bytes(data, 'utf-8'),
		headers={ "User-Agent": "Python chimerax-bugs@cgl.ucsf.edu" })
	try:
		response = urlopen(request)
	except HTTPError as e:
		from chimerax.core.errors import NonChimeraError
		raise NonChimeraError("Error from UniProt web server: %s\n\n"
			"Try again later.  If you then still get the error, you could use"
			" Help->Report a Bug to report the error to the ChimeraX team."
			" They may be able to help you work around the problem." % e)
	page = response.read().decode('utf-8')
	if not page:
		raise InvalidAccessionError("Invalid UniProt entry name / accession number: %s" % ident)
	lines = page.splitlines()
	return lines[1].split()[0]

def fetch_uniprot_accession_info(session, accession, ignore_cache=False):
	session.logger.status("Fetch UniProt accession code %s..." % accession)
	from chimerax.core.fetch import fetch_file
	name = "%s.xml" % accession
	file_name = fetch_file(session, "https://www.uniprot.org/uniprot/%s.xml" % accession,
		"%s UniProt info" % accession, name, "UniProt", ignore_cache=ignore_cache)

	session.logger.status("Parsing %s" % name)
	import xml.dom.minidom
	tree = xml.dom.minidom.parse(file_name)
	get_child = lambda parent, tag_name: [cn for cn in parent.childNodes
		if getattr(cn, "tagName", None) == tag_name][0]
	try:
		uniprot = get_child(tree, "uniprot")
	except IndexError:
		raise InvalidAccessionError("Invalid UniProt accession number: %s" % accession)

	entry = get_child(uniprot, "entry")
	try:
		seq_node = get_child(entry, "sequence")
	except (KeyError, IndexError):
		raise AssertionError("No sequence for accession %s in UniProt info" % accession)

	protein = get_child(entry, "protein")
	rec_name = [cn for cn in protein.childNodes
		if getattr(cn, "tagName", None) in ("recommendedName", "submittedName")][0]
	full_name = get_child(rec_name, "fullName").firstChild.nodeValue
	features = [cn for cn in entry.childNodes if getattr(cn, "tagName", None) == "feature"]
	return "".join([c for c in seq_node.firstChild.nodeValue if not c.isspace()]), full_name, features
