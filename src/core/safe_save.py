# vi: set expandtab shiftwidth=4 softtabstop=4:
"""
safe_save: safely write files
=============================

This module provides a method to safely overwrite a file.  If it fails,
then the file was not overwritten.

Usage:

    with SafeTextFile(filename) as f:
        print(..., file=f)
        f.write(...)

or:

    try:
        f = SafeTextFile(filename)
        print(..., file=f)
        f.write(...)
        f.close()
    except IOError as e:
        f.close(e)

"""
import os


class SafeFile:
    """Provide a file-like object to safely overwrite existing files.

    Data is first written to a temporary file, then that file is renamed to
    the desired filename when it is closed.  That way, a partial write of
    the replacement file will not overwrite the original complete file.  If
    used in a with statement, then the temporary file will always be removed
    on failure.  Defaults to writing binary files.  If no encoding is given
    for a text file, then the UTF-8 encoding is assumed.
    Locking is not provided.

    Parameters
    ----------
    filename : str
        Name of file.
    mode : string, optional
        File mode, should be 'w' or 'wb'.
    encoding : str, optional
        Text file encoding (default is UTF-8)
    critical : bool, optional
        If critical, have operating system flush to disk before closing file.

    Attributes
    ----------
    name : str
        Name of file.
    """

    def __init__(self, filename, mode='wb', encoding=None, critical=False):
        assert('w' in mode)
        if 'b' not in mode and encoding is None:
            encoding = 'utf-8'
        save_dir = os.path.dirname(filename)
        if save_dir and not os.path.isdir(save_dir):
            import errno
            raise OSError(errno.ENOTDIR, os.strerror(errno.ENOTDIR), save_dir)
        self.name = filename
        self.mode = mode
        self._critical = critical
        self._tmp_filename = filename + ".tmp"
        self._f = open(self._tmp_filename, mode, encoding=encoding)

    def __enter__(self):
        return self._f

    def __exit__(self, exc_type, exc_value, traceback):
        if not self._f.closed:
            if self._critical:
                self._f.flush()
                os.fsync(self._f)
            self._f.close()

        if self._tmp_filename is None:
            return

        if exc_type is not None:
            if os.path.exists(self._tmp_filename):
                os.unlink(self._tmp_filename)
            self._tmp_filename = None
            return

        try:
            os.rename(self._tmp_filename, self.name)
        except Exception:
            os.remove(self._tmp_filename)
            raise
        finally:
            self._tmp_filename = None

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
        self._f.write(buf)

    def writelines(self, lines):
        """Forward writing to temporary file"""
        self._f.writelines(lines)


class SafeBinaryFile(SafeFile):
    """SafeFile specialized for Binary files"""

    def __init__(self, filename, critical=False):
        SafeFile.__init__(self, filename, 'wb', critical=critical)


class SafeTextFile(SafeFile):
    """SafeFile specialized for Text files"""

    def __init__(self, filename, encoding=None, critical=False):
        SafeFile.__init__(self, filename, 'w', encoding, critical)

if __name__ == '__main__':
    testfile = 'testfile.test'

    def check_contents(contents):
        f = open(testfile)
        data = f.read()
        assert(data == contents)
        f.close()

    if os.path.exists(testfile):
        print('testfile:', testfile, 'already exists')
        raise SystemExit(1)
    try:
        # create testfile with initial contents
        try:
            f = open(testfile, 'w')
            f.write('A')
            f.close()
        except Exception as e:
            print('unable to create testfile:', testfile)
            raise SystemExit(1)
        check_contents('A')

        # overwrite the testfile
        with SafeSaveFile(testfile, 'w') as f:
            check_contents('A')
            f.write('B')
            f.flush()
            check_contents('A')
        check_contents('B')

        # try to overwrite the filefile, but fail
        try:
            with SafeSaveFile(testfile, 'w') as f:
                check_contents('B')
                f.write('A')
                f.flush()
                raise RuntimeError("fail")
        except:
            pass
            # print('successfully failed')
        assert(not os.path.exists(testfile + '.tmp'))
        check_contents('B')

    finally:
        if os.path.exists(testfile):
            os.unlink(testfile)
