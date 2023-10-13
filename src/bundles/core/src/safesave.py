# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

"""
safesave: Safely write files
============================

This module provides a method to safely overwrite a file.  If it fails,
then the file was not overwritten.

Usage::

    with SaveTextFile(filename) as f:
        print(..., file=f)
        f.write(...)

or::

    try:
        f = SaveTextFile(filename)
        print(..., file=f)
        f.write(...)
        f.close()
    except IOError as e:
        f.close(e)

"""
import os


class SaveFile:
    """Provide a file-like object to safely overwrite existing files.

    Data is first written to a temporary file, then that file is renamed to
    the desired filename when it is closed.  That way, a partial write of
    the replacement file will not overwrite the original complete file.  If
    used in a with statement, then the temporary file will always be removed
    on failure.  Defaults to writing binary files.
    Locking is not provided.

    Parameters
    ----------
    filename : str
        Name of file.
    open : function taking a filename to write
    critical : bool, optional
        If critical, have operating system flush to disk before closing file.

    Attributes
    ----------
    name : str
        Name of file.
    """

    def __init__(self, filename, open=open, critical=False):
        save_dir = os.path.dirname(filename)
        if save_dir and not os.path.isdir(save_dir):
            import errno
            raise OSError(errno.ENOTDIR, os.strerror(errno.ENOTDIR), save_dir)
        self.name = filename
        self._critical = critical
        # prevent conflicts between ChimeraX's running/starting at the same time
        self._tmp_filename = filename + "." + str(os.getpid()) + ".tmp"
        self._f = open(self._tmp_filename)
        assert(self._f.writable())

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
            try:
                os.replace(self._tmp_filename, self.name)
            except PermissionError as e:
                import sys, time
                if sys.platform.startswith("win") and e.winerror in [5, 32]:
                    # Windows can have the file open from another process
                    # for some reason (virus scan?), so wait a little and
                    # try again a few times
                    for i in range(3):
                        time.sleep(0.5)
                        try:
                            os.replace(self._tmp_filename, self.name)
                        except PermissionError:
                            if i == 2:
                                raise
                        break
                else:
                    raise
        except Exception:
            os.remove(self._tmp_filename)
            raise
        finally:
            self._tmp_filename = None

    def close(self, exception=None):
        """Close temporary file and rename it to desired filename

        If there is an exception, don't overwrite the file."""
        if exception is None:
            self.__exit__(None, None, None)
        else:
            self.__exit__(type(exception), exception, None)

    def writable(self):
        """Only writable files are supported"""
        return True

    def write(self, buf):
        """Forward writing to temporary file"""
        self._f.write(buf)

    def writelines(self, lines):
        """Forward writing to temporary file"""
        self._f.writelines(lines)


class SaveBinaryFile(SaveFile):
    """SaveFile specialized for Binary files

    Parameters
    ----------
    filename : str
        Name of file.
    critical : bool, optional
        If critical, have operating system flush to disk before closing file.
    """

    def __init__(self, filename, critical=False):
        def open_binary(filename):
            return open(filename, 'wb')
        SaveFile.__init__(self, filename, open=open_binary, critical=critical)


class SaveTextFile(SaveFile):
    """SaveFile specialized for Text files"

    Parameters
    ----------
    filename : str
        Name of file.
    encoding : str, optional
        Text file encoding (default is UTF-8)
    newline : :py:func:`open`'s optional newline argument
    critical : bool, optional
        If critical, have operating system flush to disk before closing file.
    """

    def __init__(self, filename, newline=None, encoding=None, critical=False):
        if encoding is None:
            encoding = 'utf-8'

        def open_text(filename, newline=newline, encoding=encoding):
            return open(filename, 'w', newline=newline, encoding=encoding)
        SaveFile.__init__(self, filename, open=open_text, critical=critical)

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
        with SaveTextFile(testfile, 'w') as f:
            check_contents('A')
            f.write('B')
            f.flush()
            check_contents('A')
        check_contents('B')

        # try to overwrite the file, but fail
        try:
            with SaveTextFile(testfile, 'w') as f:
                check_contents('B')
                f.write('A')
                f.flush()
                raise RuntimeError("fail")
        except Exception:
            pass
            # print('successfully failed')
        assert(not os.path.exists(testfile + '.tmp'))
        check_contents('B')

    finally:
        if os.path.exists(testfile):
            os.unlink(testfile)
