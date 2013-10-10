"""
io: manage file formats that can be opened and saved
====================================================

The io module keeps track of the functions that can open, fetch, and save
data in various formats.

I/O sources and destinations are specified as filenames, and the appropriate
open or save function is found by deducing the format from the suffix of the
filename.  An additional compression suffix, i.e., .gz, indicates that the
file is or should be compressed.  In addition to reading data from files,
data can be fetched from the Internet.  In that case, instead of a filename,
the data source is specified as prefix:identifier, e.g., pdb:1gcn, where
the prefix identifies the data format, and the identifier selects the data.

All data I/O is in binary.
"""

__all__ = [
	'register_format',
	'register_open',
	'register_fetch',
	'register_save',
	'register_compression',
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
	'requires_seeking',
	'dangerous',
	'category',
	'format_names',
	'fetch_format',
	'categorized_formats',
	'deduce_format',
	'compression_suffixes',
]

_compression = {}
def register_compression(suffix, stream_type):
	_compression[suffix] = stream_type

def _init_compression():
	try:
		import gzip
		register_compression('.gz', gzip.GzipFile)
	except ImportError:
		pass
	try:
		import bz2
		register_compression('.bz2', bz2.BZ2File)
	except ImportError:
		pass
	try:
		import lzma
		register_compression('.xz', lzma.LZMAFile)
	except ImportError:
		pass
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

class _FileFormatInfo:
	"""Keep tract of information about various data sources

	..attribute:: category

	    Type of data (STRUCTURE, SEQUENCE, etc.)

	..attribute:: extensions
	
		Sequence of filename extensions in lowercase
		starting with period (or empty)

	..attribute:: prefixes
	
		sequence of URL-style prefixes (or empty)

	..attribute:: mime_types
	
		sequence of associated MIME types (or empty)

	..attribute:: reference
	
		URL reference to specification

	..attribute:: dangerous
	
		True if can execute arbitrary code (e.g., scripts)

	..attribute:: open_func
	
		function that opens files: func(stream, identify_as=None)

	..attribute:: requires_seekable
	
		True if open function needs seekable files

	..attribute:: fetch_func
	
		function that opens internet files: func(prefixed_name, identify_as=None)

	..attribute:: save_func
	
		function that saves files: func(stream)

	..attribute:: save_notes
	
		additional information to show in save dialogs
	"""

	def __init__(self, category, extensions, prefixes, mime, reference, dangerous):
		self.category = category
		self.extensions = extensions
		self.prefixes = prefixes
		self.mime = mime
		self.reference = reference
		self.dangerous = dangerous

		self.open_func = None
		self.requires_seekable = False
		self.fetch_func = None
		self.save_func = None
		self.save_notes = None

_file_formats = {}

#TODO: _triggers = triggerSet.TriggerSet()
#TODO: NEW_FILE_FORMAT = "new file format"
#TODO: _triggers.addTrigger(NEWFILEFORMAT)

def register_format(name, category, extensions, prefixes=(), mime=(), reference=None, dangerous=None, **kw):
	"""Register file format's I/O functions and meta-data

	:param name: format's name
	:param category: says what kind of data the should be classified as.
	:param extensions: is a sequence of filename suffixes starting
	   with a period.  If the format doesn't open from a filename
	   (e.g., PDB ID code), then extensions should be an empty sequence.
	:param prefixes: is a sequence of filename prefixes (no ':'),
	   possibily empty.
	:param mime: is a sequence of mime types, possibly empty.
	:param reference: a URL link to the specification. 
	:param dangerous: should be True for formats that can write/delete
	   a users's files.  False by default except for the SCRIPT category.

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
	ff = _file_formats[name] = _FileFormatInfo(category, exts, prefixes,
			mime, reference, dangerous)
	for attr in ['open_func', 'requires_seekable', 'fetch_func',
			'save_func', 'save_notes']:
		if attr in kw:
			setattr(ff, attr, kw[attr])
	#TODO: _triggers.activateTrigger(NEW_FILE_FORMAT, name)

def prefixes(name):
	"""Return filename prefixes for named format.

	prefixes(name) -> [filename-prefix(es)]
	"""
	try:
		return _file_formats[name].prefixes
	except KeyError:
		return ()

def register_open(name, open_function, requires_seeking=False):
	"""register a function that reads data from a stream

	:param open_function: function taking an I/O stream
	:param requires_seeking: True if stream must be seekable
	"""
	try:
		fi = _file_formats[name]
	except KeyError:
		raise ValueError("Unknown data type")
	fi.open_func = open_function
	fi.requires_seeking = requires_seeking

def register_fetch(name, fetch_function):
	"""register a function that fetches data from the Internet

	:param fetch_fuction: function that takes an identifier,
	    and returns an I/O stream for reading data.
	"""
	try:
		fi = _file_formats[name]
	except KeyError:
		raise ValueError("Unknown data type")
	fi.fetch_func = fetch_function

def register_save(name, save_function, save_notes=''):
	try:
		fi = _file_formats[name]
	except KeyError:
		raise ValueError("Unknown data type")
	fi.save_func = save_function
	fi.save_notes = save_notes

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

def requires_seeking(name):
	"""Return whether named format can needs a seekable file"""
	try:
		return _file_formats[name].requires_seeking
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

def format_names(open=True, save=False, source_is_file=False):
	"""Return known format names.

	formats() -> [format-name(s)]
	"""
	names = []
	for t, info in _file_formats.items():
		if open and not info.open_func:
			continue
		if save and not info.save_func:
			continue
		if not source_is_file or info.extensions:
			names.append(t)
	return names

def categorized_formats(open=True, save=False):
	"""Return known formats by category

	categorized_formats() -> { category: formats() }
	"""
	result = {}
	for name, info in _file_formats.items():
		if open and not info.open_func:
			continue
		if save and not info.save_func:
			continue
		names = result.setdefault(info.category, [])
		names.append(name)
	return result

def initialize_formats():
	# TODO: have set of directories to look for formats in?
	from . import formats
	formats.initialize()

def deduce_format(filename, default_name=None, prefixable=True):
	"""Figure out named format associated with filename
	
	Return tuple of deduced format, whether it was a prefix reference,
	the unmangled filename, and the compression format (if present).
	If it is a prefix reference, then it needs to be fetched.
	"""
	name = None
	prefixed = False
	compression = None
	if prefixable:
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
		for compression in compression_suffixes():
			if filename.endswith(compression):
				stripped = filename[:-len(compression)]
				break
		else:
			stripped = filename
			compression = None
		base, ext = os.path.splitext(stripped)
		ext = ext.lower()
		for t, info in _file_formats.items():
			if ext in info.extensions:
				name = t
				break
		if name == None:
			name = default_name
	return name, prefixed, filename, compression

def qt_save_file_filter(category=None, all=False):
	"""Return file name filter suitable for Save File dialog"""

	result = []
	for t, info in _file_formats.items():
		if not info.save_func:
			continue
		if category and info.category != category:
			continue
		exts = ' '.join('*%s' % ext for ext in info.extensions)
		result.append("%s files (%s)" % (t, exts))
	if all:
		result.append("All files (*)")
	result.sort(key=str.casefold)
	return ';;'.join(result)

def qt_open_file_filter(all=False):
	"""Return file name filter suitable for Open File dialog"""

	combine = {}
	for t, info in _file_formats.items():
		if not info.open_func:
			continue
		exts = combine.setdefault(info.category, [])
		exts.extend(info.extensions)
	result = ["%s files (%s)" % (k, ' '.join('*%s' % ext for ext in combine[k])) for k in combine]
	if _compression:
		result.append("Compressed files (%s)" % ' '.join(_compression.keys()))
	if all:
		result.append("All files (*)")
	result.sort(key=str.casefold)
	return ';;'.join(result)

_builtin_open = open
def open(filespec, identify_as=None, **kw):
	"""open a (compressed) file
	
	:param filespec: '''prefix:id''' or a (compressed) filename
	:param identify_as: name used to identify data source; default to filespec

	If a file format needs a seekable file, then compressed files are
	uncompressed into a temporary file before calling the open function.
	"""

	from chimera2.cmds import UserError
	name, prefix, filelike, compression = deduce_format(filespec)
	if not identify_as:
		identify_as = filespec
	if name is None:
		raise UserError("Missing or unknown file type")
	open_func = open_function(name)
	if open_func is None:
		raise UserError("unable to open %s files" % name)
	if prefix:
		fetch = fetch_function(name)
		if fetch is None:
			raise UserError("unable to fetch %s files" % name)
		stream = fetch(filelike)
	else:
		if not compression:
			import os
			filename = os.path.expanduser(os.path.expandvars(filelike))
			try:
				stream = _builtin_open(filename, 'rb')
			except OSError as e:
				raise UserError(e)
		else:
			stream_type = _compression[name]
			try:
				stream = stream_type(filelike)
			except OSError as e:
				raise UserError(e)
			if requires_seeking(name):
				# copy compressed file to real file
				import tempfile
				exts = extensions(name)
				suffix = exts[0] if exts else ''
				tf = tempfile.NamedTemporaryFile(prefix='chtmp', suffix=suffix)
				while 1:
					data = stream.read()
					if not data:
						break
					tf.write(data)
				tf.seek(0)
				stream = tf
	return open_func(stream, identify_as=identify_as, **kw)

def save(filename, **kw):
	from chimera2.cmds import UserError
	name, prefix, filelike, compression = deduce_format(filename, prefixable=False)
	if name is None:
		raise UserError("Missing or unknown file type")
	func = save_function(name)
	if not compression:
		stream = open(filelike, 'wb')
	else:
		stream_type = _compression[name]
		stream = stream_type(filelike)
	return func(stream, **kw)
