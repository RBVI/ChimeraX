"""
data: manage information about file formats that can be opened and saved
========================================================================

.. Note:

   This module's interface will change.

The data module keeps track of the functions that can open and save
various data formats.
The functions do I/O to a named file or, optionally, to a file stream.

.. todo:

   If file streams are supported,
   then various registered compression schemes are available as well.
"""

__all__ = [
	'register_format',
	'DEFAULT_CATEGORY',
	'DYNAMICS',
	'GENERIC3D',
	'SCRIPT',
	'SEQUENCE',
	'STRUCTURE',
	'SURFACE',
	'VOLUME',
	'open',
	'prefixes',
	'extensions',
	'open_function',
	'fetch_function',
	'save_function',
	'mime_types',
	'can_decompress',
	'dangerous',
	'category',
	'formats',
	'fetch_format',
	'categorized_formats',
	'deduce_format',
	'compression_suffixes',
]


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
DEFAULT_CATEGORY = "Miscellaneous"	#: default file format category
DYNAMICS = "Molecular trajectory"	#: trajectory
GENERIC3D = "Generic 3D object"
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


def register_format(name, open_function, fetch_function, save_function,
		extensions, prefixes,
		mime=(), can_decompress=True, dangerous=None,
		category=DEFAULT_CATEGORY, fetch_format=None,
		reference=None, save_notes=None):
	"""Register file format's I/O functions and meta-data

	:param name: format's name
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
		print("missing fetch function for format with prefix support:", name, file=sys.stderr)
	if mime is None:
		mime = ()
	_file_formats[name] = _FileFormatInfo(
			open_function, fetch_function, save_function,
			exts, prefixes, mime,
			can_decompress, dangerous, category,
			fetch_format, reference, save_notes)
	#TODO: _triggers.activateTrigger(NEW_FILE_FORMAT, name)

def prefixes(name):
	"""Return filename prefixes for named format.

	prefixes(name) -> [filename-prefix(es)]
	"""
	try:
		return _file_formats[name].prefixes
	except KeyError:
		return ()

def extensions(name):
	"""Return filename extensions for named format.

	extensions(name) -> [filename-extension(s)]
	"""
	try:
		exts = _file_formats[name].extensions
	except KeyError:
		return ()
	return exts

def open_function(name):
	"""Return open callback for named format.

	open_function(name) -> function
	"""
	try:
		return _file_formats[name].open_func
	except KeyError:
		return None

def fetch_function(name):
	"""Return fetch callback for named format.

	fetch_function(name) -> function
	"""
	try:
		return _file_formats[name].fetch_func
	except KeyError:
		return None

def save_function(name):
	"""Return save callback for named format.

	save_function(name) -> function
	"""
	try:
		return _file_formats[name].save_func
	except KeyError:
		return None

def mime_types(name):
	"""Return mime types for named format."""
	try:
		return _file_formats[name].mime_types
	except KeyError:
		return None

def can_decompress(name):
	"""Return whether named format can open compressed files"""
	try:
		return _file_formats[name].can_decompress
	except KeyError:
		return False

def dangerous(name):
	"""Return whether named format can write to files"""
	try:
		return _file_formats[name].dangerous
	except KeyError:
		return False

def category(name):
	"""Return category of named format"""
	try:
		return _file_formats[name].category
	except KeyError:
		return "Unknown"

def formats(source_is_file=False):
	"""Return known format names.

	formats() -> [format-name(s)]
	"""
	if source_is_file:
		formats = []
		for t, info in _file_formats.items():
			if info.extensions:
				formats.append(t)
		return formats
	return _file_formats.keys()

def fetch_format(name):
	"""Return format name to display when fetched via the web"""
	try:
		return _file_formats[name].fetch_format or name
	except KeyError:
		return name

def categorized_formats():
	"""Return known formats by category

	categorized_formats() -> { category: formats() }
	"""
	result = {}
	for name, info in _file_formats.items():
		formats = result.setdefault(info.category, [])
		formats.append(name)
	return result

def deduce_format(filename, default_format=None, prefixable_format=True):
	"""Figure out named format associated with filename
	
	Return tuple of deduced format, whether it was a prefix reference,
	and the unmangled filename.  If it is a prefix reference, then
	it needs to be fetched."""
	name = None
	prefixed = False
	if prefixable_format:
		# format may be specified as colon-separated prefix
		try:
			prefix, fname = filename.split(':', 1)
		except ValueError:
			pass
		else:
			for t, info in _file_formats.items():
				if prefix in info.prefixes:
					name = t
					filename = fname
					prefixed = True
					break
	if name == None:
		import os
		for cs in compression_suffixes():
			if filename.endswith(cs):
				stripped, ext = os.path.splitext(filename)
				break
		else:
			stripped = filename
		base, ext = os.path.splitext(stripped)
		ext = ext.lower()
		for t, info in _file_formats.items():
			if ext in info.extensions:
				name = t
				break
		if name == None:
			name = default_format
	return name, prefixed, filename

def qt_save_file_filter(category=None, all=False):
	"""Return file name filter suitable for Save File dialog"""

	result = []
	for t, info in _file_formats.items():
		if category and info.category != category:
			continue
		exts = ' '.join('*%s' % ext for ext in info.extensions)
		result.append("%s files (%s)" % (t, exts))
	if all:
		result.append("All files (*)")
	result.sort(key=str.casefold)
	return ';;'.join(result)

def qt_open_file_filter(all=False, expand=False):
	"""Return file name filter suitable for Open File dialog"""

	if expand:
		return qt_save_file_filter(all=all)

	combine = {}
	for t, info in _file_formats.items():
		exts = combine.setdefault(info.category, [])
		exts.extend(info.extensions)
	result = ["%s files (%s)" % (k, ' '.join('*%s' % ext for ext in combine[k])) for k in combine]
	if all:
		result.append("All files (*)")
	result.sort(key=str.casefold)
	return ';;'.join(result)

def open(filename, identify_as=None):
	name, prefix, filename = deduce_format(filename)
	if prefix:
		func = fetch_function(name)
	else:
		func = open_function(name)
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
