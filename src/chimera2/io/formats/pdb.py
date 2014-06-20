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

def open_mmCIF(stream, *args, **kw):

	if hasattr(stream, 'name'):
		filename = stream.name
	else:
		# it's really a filename
		filename = stream

	import pdbio
	import os, mmap
	fd = os.open(filename, os.O_RDONLY)
	st = os.stat(fd)
	# need a zero byte terminated buffer for parse_mmCIF_file
	if os.name == 'posix' and (st.st_size % mmap.PAGESIZE) == 0:
		# mmap with files that are a multiple of the pagesize
		# cause a bus error on POSIX systems
		buf = os.read(fd, st.st_size)
		mol_blob = pdbio.parse_mmCIF_file(buf)
	try:
		if os.name == 'posix':
			# not a multiple of pagesize, rest of page
			# is zero filled
			size = st.st_size
		else:
			# Windows, ask for an extra byte to guarantee
			# termination with zero byte
			size = st.st_size + 1
		with mmap.mmap(fd, size, access=mmap.ACCESS_READ) as buf:
			mol_blob = pdbio.parse_mmCIF_file(buf)
	finally:
		os.close(fd)
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
        # mmCIF uses same file suffix as CIF
	#io.register_format("CIF", molecule.CATEGORY, (".cif",), ("cif", "cifID"),
	#	mime=("chemical/x-cif"),
	#	reference="http://www.iucr.org/__data/iucr/cif/standard/cifstd1.html")
	io.register_format("mmCIF", molecule.CATEGORY, (".cif",), ("mmcif", "cif"),
		mime=("chemical/x-cif", "chemical/x-mmcif"),
		reference="http://mmcif.rcsb.org/",
		requires_seeking=True, open_func=open_mmCIF)
