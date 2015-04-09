# vi: set expandtab shiftwidth=4 softtabstop=4:
"""
safe_save: safely write files
=============================

This module provides a method to safely overwrite a file.  If it fails,
then the file was not overwritten.

Usage:

    with SafeSaveFile(filename) as f:
        print(..., file=f)
        f.write(...)

or:

    try:
        f = SafeSave(filename)
        print(..., file=f)
        f.write(...)
        f.close()
    except IOError as e:
        f.close(e)

"""
import os


class SafeSaveFile:
    """Provide a file-like object to safely overwrite existing files.

    Data is first written to a temporary file, then that file is renamed to
    the desired filename when it is closed.  That way, a partial write of
    the replacement file will not overwrite the original complete file.  If
    used in a with statement, then the temporary file will always be removed
    on failure.  Defaults to writing binary files.  If no encoding is given
    for a text file, then the UTF-8 encoding is assumed.
    Locking is not provided.
    """

    def __init__(self, filename, mode='wb', encoding=None):
        assert('w' in mode)
        if 'b' not in mode and encoding is None:
            encoding = 'utf-8'
        save_dir = os.path.dirname(filename)
        if not os.path.isdir(save_dir):
            import errno
            raise OSError(errno.ENOTDIR, os.strerror(errno.ENOTDIR), save_dir)
        self.filename = filename
        self.tmp_filename = filename + ".tmp"
        self.mode = mode
        self.f = open(self.tmp_filename, mode, encoding=encoding)

    def __enter__(self):
        return self.f

    def __exit__(self, exc_type, exc_value, traceback):
        if not self.f.closed:
            self.f.flush()
            os.fsync(self.f)
            self.f.close()

        if exc_type is not None:
            if os.path.exists(self.tmp_filename):
                os.unlink(self.tmp_filename)
            self.tmp_filename = None
            return

        if self.tmp_filename is None:
            return

        try:
            os.rename(self.tmp_filename, self.filename)
        finally:
            os.remove(self.tmp_filename)
            self.tmp_filename = None

    def close(self, exception=None):
        """Close temporary file and rename it to desired filename"""
        if exception is None:
            self.__exit__(None, None, None)
        else:
            self.__exit__(type(exception), exception, None)

    def writeable(self):
        """Only writeable files are supported"""
        return True

    def write(self, buf):
        """Forward writing to temporary file"""
        self.f.write(buf)

    def writelines(self, lines):
        """Forward writing to temporary file"""
        self.f.writelines(lines)
