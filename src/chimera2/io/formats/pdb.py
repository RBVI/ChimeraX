from chimera2 import molecule
from chimera2.cli import UserError

_builtin_open = open

def open_pdb(stream, *args, **kw):

	if hasattr(stream, 'read'):
		input = stream
	else:
		# it's really a filename
		input = _builtin_open(stream, 'rb')

	import pdbio
	mol_blob = pdbio.read_pdb_file(input)
	if input != stream:
		input.close()
	model = molecule.Molecule()
	model.mol_blob = mol_blob
	model.make_graphics()

	return [model], "Model %d: PDB data containing %d atoms and %d bonds" % (model.id, model.num_atoms, model.num_bonds)

def fetch_pdb(pdb_id):
	if len(pdb_id) != 4:
		raise UserError("PDB identifiers are 4 characters long")
	import os
	# check in local cache
	filename = "~/Downloads/Chimera/PDB/%s.pdb" % pdb_id.upper()
	filename = os.path.expanduser(filename)
	if os.path.exists(filename):
		return _builtin_open(filename, 'rb')
	# check on local system -- TODO: configure location
	filename = "/databases/mol/pdb/pdb%s.ent" % pdb_id.lower()
	if os.path.exists(filename):
		return _builtin_open(filename, 'rb')
	from urllib.request import URLError, Request, urlopen
	url = "http://www.rcsb.org/pdb/files/%s.pdb" % pdb_id.upper()
	# TODO: save in local cache
	request = Request(url, headers={
		# TODO: get application name and version for User-Agent
		"User-Agent": "UCSF-Chimera2/0.1",
	})
	try:
		return urlopen(request)
	except URLError as e:
		raise UserError(str(e))

def open_mmcif(stream, *args, **kw):

	if hasattr(stream, 'name'):
		filename = stream.name
	else:
		# it's really a filename
		filename = stream

	import pdbio
	mol_blob = pdbio.read_mmcif_file(filename)
	model = molecule.Molecule()
	model.mol_blob = mol_blob
	model.make_graphics()

	return [model], "Model %d: PDB data containing %d atoms and %d bonds" % (model.id, model.num_atoms, model.num_bonds)

def register():
	from chimera2 import io
	io.register_format("PDB", molecule.CATEGORY,
		(".pdb", ".pdb1", ".ent", ".pqr"), ("pdb",),
		mime=("chemical/x-pdb", "chemical/x-spdbv"),
		reference="http://wwpdb.org/docs.html#format",
		open_func=open_pdb, fetch_func=fetch_pdb)
	# Don't register CIF separately since mmCIF also uses .cif suffix
	#io.register_format("CIF", molecule.CATEGORY, (".cif",), ("cif", "cifID"),
	#	mime=("chemical/x-cif"),
	#	reference="http://www.iucr.org/__data/iucr/cif/standard/cifstd1.html")
	io.register_format("mmCIF", molecule.CATEGORY, (".mcif", ".cif"), ("mmcif",),
		mime=("chemical/x-mmcif", "chemical/x-cif"),
		reference="http://mmcif.rcsb.org/",
		requires_seeking=True, open_func=open_mmcif)
