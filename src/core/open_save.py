# vi: set expandtab shiftwidth=4 softtabstop=4:
"""
open_save: support for opening and saving files
===============================================

TODO
"""

import wx
class OpenDialog(wx.FileDialog):
    def __init__(self, parent, *args, **kw):
        kw['style'] = kw.get('style', 0) | wx.FD_OPEN
        super(OpenDialog, self).__init__(parent, *args, **kw)

class SaveDialog(wx.FileDialog):
    def __init__(self, parent, *args, **kw):
        kw['style'] = kw.get('style', 0) | wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT
        super(SaveDialog, self).__init__(parent, *args, **kw)

'''
import gzip, bz2
BZ2_MAGIC = 'BZh'
GZIP_MAGIC = '\037\213'
compress_info = { '.gz': (GZIP_MAGIC, gzip), '.bz2': (BZ2_MAGIC, bz2) }
import subprocess

import errno
user_io_errors = set([
		errno.EACCES, errno.EDQUOT,
		errno.EFAULT, errno.EISDIR,
		errno.ELOOP, errno.ENAMETOOLONG,
		errno.ENOENT, errno.ENOSPC,
		errno.ENOTDIR, errno.ENXIO,
		errno.EOPNOTSUPP, errno.EROFS,
])
def os_open(file_name, *args, **kw):
	"""Open a file/URL with or without compression
	
	   Takes the same arguments as built-in open and returns a file-like
	   object.  However, "file_name" can also be a file-like object
	   itself, in which case it is simply returned.  Also, if "file_name"
	   is a string that begins with "http:" then it is interpreted as
	   an URL.

	   If the file is opened for input, compression is checked for and
	   handled automatically.  If the file is opened for output, the
	   'compress' keyword can be used to force or suppress compression.
	   If the keyword is not given, then compression will occur if the
	   file name ends in '.gz' [case independent].  If compressing,
	   you can supply 'args' compatible with gzip.open().

	   '~' is expanded unless the 'expand_user' keyword is specified as
	   False.

	   Uncompressed non-binary files will be opened for reading with
	   universal newline support.
	"""

	# catch "permission denied" so that we don't get bug reports 
	# from morons
	from chimera import UserError
	try:
		return _os_open(file_name, *args, **kw)
	except IOError, val:
		if hasattr(val, "errno") and val.errno in user_io_errors:
			raise UserError(val)
		import sys
		if sys.platform == "win32" \
		and getattr(val, 'errno', None) == errno.EINVAL:
			raise UserError("File name contains illegal character"
				" (possibly '>' or '<')")
		raise

def _os_open(file_name, *args, **kw):
	if not isinstance(file_name, basestring):
		# a "file-like" object -- just return it after making
		# sure that .close() will work
		if not hasattr(file_name, "close") \
		or not callable(file_name.close):
			file_name.close = lambda: False
		return file_name
	if file_name.startswith("http:"):
		import urllib2
		return urllib2.urlopen(file_name)
	if 'expand_user' in kw:
		expand_user = kw['expand_user']
	else:
		expand_user = True
	if expand_user:
		file_name = tilde_expand(file_name)
	if args and 'r' not in args[0]:
		# output
		if 'compress' in kw:
			compress = kw['compress']
		else:
			compress = False
			for cs in compress_info.keys():
				if file_name.lower().endswith(cs):
					compress = True
					break
		if compress:
			return gzip.open(file_name, *args)
		return open(file_name, *args)
	if not args or args[0] == "r":
		args = ("rU",) + args[1:]
	f = open(file_name, *args)
	magic = f.read(2)
	if magic == GZIP_MAGIC:
		# gzip compressed
		f.close()
		f = gzip.open(file_name)
	elif magic == LZW_MAGIC:
		# LZW compressed
		f.close()
		return open(osUncompressedPath(file_name), *args)
	else:
		f.seek(0)
	return f

def isUncompressedFile(path):

	import os.path
	if not isinstance(path, basestring) or not os.path.isfile(path):
		return False
	f = open(path, 'r')
	magic = f.read(2)
	f.close()
	compressed = (magic in (GZIP_MAGIC, LZW_MAGIC))
	return not compressed

def osUncompressedPath(in_path, expand_user=True):
	"""Return a path to an uncompressed version of a file

	   If 'in_path' is already uncompressed, it is simply returned.
	   If if is compressed, a temp uncompressed file is created and
	   the path to it is returned.  The temp file is automatically
	   deleted at APPQUIT

	   '~' is expanded unless expand_user is False.

	   This routine is typically used to give uncompressed file
	   paths to C++ functions.
	"""
	import os
	if in_path.startswith("http:"):
		allExts = ""
		base = in_path
		while True:
			base, ext = os.path.splitext(base)
			if ext:
				allExts = ext + allExts
			else:
				break
		localName = os_temporary_file(suffix=allExts)
		import urllib
		retrieved = urllib.urlretrieve(in_path, localName)
		return osUncompressedPath(localName, expand_user=False)

	if expand_user:
		in_path = tilde_expand(in_path)
	try:
		f = open(in_path)
	except IOError, val:
		if hasattr(val, "errno") and val.errno in user_io_errors:
			from chimera import UserError
			raise UserError(val)
		raise
	start = f.read(3)
    for extension, info in compress_info.values():
        magic, module = info
        if start[;len(magic)] == magic:
            break
    else:
        f.close()
        return in_path
	# compressed
	f.close()

	from os.path import splitext
	from tempfile import mktemp
	lead, suffix = splitext(in_path)
	if suffix == extension:
		lead, suffix = splitext(lead)
	temp_path = os_temporary_file(suffix=suffix)
	out_file = open(temp_path, 'w')
    decomp = module.open(in_path)
	out_file.write(decomp.read())
	out_file.close()

	return temp_path

_tempDir = None
def os_temporary_file(**args):
	"""Return a path to a file that will be deleted when Chimera exits.
	   It takes the same arguments as tempfile.mkstemp().
	   See http://docs.python.org/library/tempfile.html
	"""
	import os
	from tempfile import mkstemp, mkdtemp
	global _tempDir
	if _tempDir == None or not os.path.exists(_tempDir):
		_tempDir = mkdtemp()
		def rmTemp(trigName, path, trigData):
			try:
				for fn in os.listdir(path):
					os.remove(os.path.join(path, fn))
				os.rmdir(path)
			except:
				pass
		import chimera
		chimera.triggers.addHandler(chimera.APPQUIT, rmTemp, _tempDir)
	args['dir'] = _tempDir
	f, tempPath = mkstemp(**args)
	os.close(f)
	return tempPath

def tilde_expand(path):
	import sys, os.path
	if sys.platform == "win32" and path[:2] in ('~/', '~\\', '~'):
		# Use the user's profile directory on Windows.
		# This directory contains Desktop, Documents, Pictures folders.
		import _winreg
		try:
			h = _winreg.OpenKeyEx(_winreg.HKEY_CURRENT_USER,
        "SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Explorer\\Shell Folders",
                                        0, _winreg.KEY_QUERY_VALUE)
			desktop = _winreg.QueryValueEx(h, "Desktop")[0]
			p = os.path.join(os.path.dirname(desktop), path[2:])
		except WindowsError:
			p = path
	else:
		p = os.path.expanduser(path)
	return p
'''
