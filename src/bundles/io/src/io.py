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

from .compression import handle_compression, get_compression_type

def open_input(source, encoding=None, *, compression=None):
	"""Open possibly compressed input for reading.
		'source' can be path or a stream.  If a stream, it is simply returned.
		If encoding is 'None', open as binary.
		If 'compression' is None, whether to use compression and what type
			will be determined off the file name."""
	if _is_stream(source):
		return source
	mode = 'rt' if encoding else 'rb'
	compression_type = get_compression_type(source, compression)
	if compression_type:
		return handle_compression(compression_type, source, mode=mode, encoding=encoding)
	return open(source, mode, encoding=encoding)

def open_output(output, encoding=None, *, compression=None):
	"""Open output for (possibly compressed) writing.
		'output' can be path or a stream.  If a stream, it is simply returned.
		If encoding is 'None', open as binary
		If 'compression' is None, whether to use compression and what type
			will be determined off the file name."""
	if _is_stream(output):
		return output
	compression_type = get_compression_type(output, compression)
	mode = 'wt' if encoding else 'wb'
	if compression_type:
		return handle_compression(compression_type, output, mode=mode, encoding=encoding)
	return open(output, mode, encoding=encoding)

def _is_stream(source):
	if isinstance(source, str):
		return False
	# ensure that 'close' works on the stream...
	if not hasattr(source, "close") or not callable(source.close):
		source.close = lambda: False
	return True
