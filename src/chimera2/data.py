"""
data: manage information about file formats that can be opened and saved
========================================================================
"""

import collections

_compression = {}
def register_compression(suffix, file_type):
	_compression[suffix] = file_type

def _init_compression():
	import gzip, bz2
	register_compression('.gz', gzip.GzipFile)
	register_compression('.bz2', bz2.BZ2File)
_init_compression()

def compression_suffixes():
	return _compression.keys()

# some well known file format categories
DEFAULT_CATEGORY = "Miscellaneous"
DYNAMICS = "Molecular trajectory"
GENERIC3D = "Generic 3D objects"
SCRIPT = "Command script"
SEQUENCE = "Sequence alignment"
STRUCTURE = "Molecular structure"
SURFACE = "Molecular surface"
VOLUME = "Volume data"

Categories = (
	DEFAULT_CATEGORY, DYNAMICS, GENERIC3D,
	SCRIPT, SEQUENCE, STRUCTURE, SURFACE,
	VOLUME,
)

_FileFormatInfo = collections.namedtuple('_FileFormatInfo', (
	'open_func',		# python function that opens files:
				#	func(filename, identify_as=None)
	'fetch_func',		# python function that opens internet files:
				#	func(prefixed_name, identify_as=None)
	'save_func',		# python function that saves files:
				#	func(filename)
	'extensions',		# sequence of filename extensions in lowercase
				#	starting with period (or empty)
	'prefixes',		# sequence of URL-style prefixes (or empty)
	'mime_types',		# sequence of associated MIME types (or empty)
	'can_decompress',	# True if open function handles compressed files
	'dangerous',		# True if can execute arbitrary code (scripts)
	'category',		# data category
	'fetch_format',		# format name to display when fetched
				# if different from format name
	'reference',		# URL reference to specification
	'save_notes',		# additional information to show in save dialogs
))
_file_formats = {}

#TODO: _triggers = triggerSet.TriggerSet()
#TODO: NEW_FILE_FORMAT = "new file format"
#TODO: _triggers.addTrigger(NEWFILEFORMAT)


def register_format(format, open_function, fetch_function, save_function,
		extensions, prefixes,
		mime=(), can_decompress=True, dangerous=None,
		category=DEFAULT_CATEGORY, fetch_format=None,
		reference=None, save_notes=None):
	"""Register file format's I/O functions and meta-data

	:param extensions: is a sequence of filename suffixes starting
	   with a period.  If the format doesn't open from a filename
	   (e.g., PDB ID code), then extensions should be an empty sequence.
	:param prefixes: is a sequence of filename prefixes (no ':'),
	   possibily empty.
	:param mime: is a sequence of mime types, possibly empty.
	:param can_decompress: set to False if the format doesn't want to
	   be given compressed files.
	:param dangerous: should be True for scripts and other formats that
	   can write/delete a users's files.
	:param category: says what kind of data the should be classified as.

	.. todo::
	    possibly break up in to multiple functions
	"""
	if dangerous is None:
		# scripts are inherently dangerous
		dangerous = category == SCRIPT
	if extensions is not None:
		exts = [s.lower() for s in extensions]
	else:
		exts = ()
	if prefixes is None:
		prefixes = ()
	if prefixes and not fetch_function:
		import sys
		print >> sys.stderr, "missing fetch function for format with prefix support:", format
	if mime is None:
		mime = ()
	_file_formats[format] = _FileFormatInfo(
			open_function, fetch_function, save_function,
			exts, prefixes, mime,
			can_decompress, dangerous, category,
			fetch_format, reference, save_notes)
	#TODO: _triggers.activateTrigger(NEW_FILE_FORMAT, format)

def prefixes(format):
	"""Return filename prefixes for given format.

	prefixes(format) -> [filename-prefix(es)]
	"""
	try:
		return _file_formats[format].prefixes
	except KeyError:
		return ()

def extensions(format):
	"""Return filename extensions for given format.

	extensions(format) -> [filename-extension(s)]
	"""
	try:
		exts = _file_formats[format].extensions
	except KeyError:
		return ()
	return exts

def open_function(format):
	"""Return open callback for given format.

	open_function(format) -> function
	"""
	try:
		return _file_formats[format].open_func
	except KeyError:
		return None

def fetch_function(format):
	"""Return fetch callback for given format.

	fetch_function(format) -> function
	"""
	try:
		return _file_formats[format].fetch_func
	except KeyError:
		return None

def save_function(format):
	"""Return save callback for given format.

	save_function(format) -> function
	"""
	try:
		return _file_formats[format].save_func
	except KeyError:
		return None

def mime_types(format):
	"""Return mime types for given format."""
	try:
		return _file_formats[format].mime_types
	except KeyError:
		return None

def can_decompress(format):
	"""Return whether this format can open compressed files"""
	try:
		return _file_formats[format].can_decompress
	except KeyError:
		return False

def dangerous(format):
	"""Return whether this format can write to files"""
	try:
		return _file_formats[format].dangerous
	except KeyError:
		return False

def category(format):
	"""Return category of this format"""
	try:
		return _file_formats[format].category
	except KeyError:
		return "Unknown"

def formats(source_is_file=False):
	"""Return known formats.

	formats() -> [format(s)]
	"""
	if source_is_file:
		formats = []
		for t, info in _file_formats.iteritems():
			if info.extensions:
				formats.append(t)
		return formats
	return _file_formats.keys()

def fetch_format(format):
	"""Return format to display when fetched via the web"""
	try:
		return _file_formats[format].fetch_format or format
	except KeyError:
		return format

def categorized_formats():
	"""Return known formats by category

	categorized_formats() -> { category: formats() }
	"""
	result = {}
	for format, info in _file_formats.iteritems():
		formats = result.setdefault(info.category, [])
		formats.append(format)
	return result

def deduce_format(filename, default_format=None, prefixable_format=True):
	"""Figure out format associated with filename
	
	Return tuple of deduced format, whether it was a prefix reference,
	and the unmangled filename.  If it is a prefix reference, then
	it needs to be fetched."""
	format = None
	prefixed = False
	if prefixable_format:
		# format may be specified as colon-separated prefix
		try:
			prefix, fname = filename.split(':', 1)
		except ValueError:
			pass
		else:
			for t, info in _file_formats.iteritems():
				if prefix in info.prefixes:
					format = t
					filename = fname
					prefixed = True
					break
	if format == None:
		import os
		for cs in compression_suffixes():
			if filename.endswith(cs):
				stripped, ext = os.path.splitext(filename)
				break
		else:
			stripped = filename
		base, ext = os.path.splitext(stripped)
		ext = ext.lower()
		for t, info in _file_formats.iteritems():
			if ext in info.extensions:
				format = t
				break
		if format == None:
			format = default_format
	return format, prefixed, filename

def open(filename, identify_as=None):
	format, prefix, filename = deduce_format(filename)
	if prefix:
		func = fetch_function(format)
	else:
		func = open_function(format)
	if func is None:
		from . import cmds
		raise cmds.UserError("unknown file type")
	return func(filename, identify_as=identify_as)

#Examples to be provided in other code:
#
#register_format("PDB",
#	_openPDBModel, None, None, (".pdb", ".pdb1", ".ent", ".pqr"), ("pdb",),
#	mime=("chemical/x-pdb", "chemical/x-spdbv"),
#	category=STRUCTURE)
#register_format("PDBID",
#	_openPDBIDModel, None, None, (), ("pdbID",),
#	category=STRUCTURE, can_decompress=False, fetch_format="PDB")
#register_format("CIFID",
#	_openCIFIDModel, None, None, (), ("cifID",),
#	category=STRUCTURE, can_decompress=False, fetch_format="mmCIF")
#register_format("VRML",
#	_openVRMLModel, None, None, (".wrl", ".vrml"), ("vrml",),
#	category=GENERIC3D, mime=("model/vrml",),
#	reference="http://www.web3d.org/x3d/specifications/#vrml97")
#register_format("X3D",
#	_openX3DModel, None, _exportX3D, (".x3d",), ("x3d",),
#	category=GENERIC3D, mime=("model/x3d+xml",),
#	reference="http://www.web3d.org/x3d/specifications/#x3d-spec",
#	save_notes="  Not supported: hither/yon clipping, per-model"
#		 " clipping planes, depth-cueing."
#		 "  Although there are annotations for everything but"
#		 " stereo.")
#register_format("Mol2",
#	_openMol2Model, None, None, (".mol2",), ("mol2",),
#	category=STRUCTURE, mime=("chemical/x-mol2",))
#register_format("Python",
#	_openPython, None, None, (".py", ".pyc", ".pyo", ".pyw"),
#	("python", "py", "chimera"),
#	category=SCRIPT, mime=("application/x-chimera",))
#register_format("Gaussian formatted checkpoint",
#	_openGaussianFCF, None, None, (".fchk",), ("fchk", "gaussian"),
#	category=STRUCTURE, mime=("chemical/x-gaussian-checkpoint",))
#register_format("SCOP",
#	_openSCOPModel, None, None, (), ("scop",), can_decompress=False,
#	category=STRUCTURE)
#register_format("NDB", None, _openNDBModel, None, (), ("ndb",), can_decompress=False,
#	category=STRUCTURE)
