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
utils: Generically useful stuff that doesn't fit elsewhere
==========================================================
"""
import os
import subprocess
import sys

from contextlib import contextmanager


# Based on Mike C. Fletcher's BasicTypes library
# https://sourceforge.net/projects/basicproperty/ and comments in
# http://rightfootin.blogspot.com/2006/09/more-on-python-flatten.html
# Except called flattened, like sorted, since it is nondestructive
def flattened(input, *, return_types=(list, tuple, set), return_type=None, maxsize=sys.maxsize):
    """Return new flattened version of input

    Parameters
    ----------
    input : a sequence instance (list, tuple, or set)
    return_type : optional return type (defaults to input type)

    Returns
    -------
    A sequence of the same type as the input.
    """
    if return_type is None:
        return_type = type(input)
        if return_type not in return_types:
            return_type = list  # eg., not zip
    output = list(input)
    try:
        # for every possible index
        for i in range(maxsize):
            # while that index currently holds a list
            while isinstance(output[i], return_types):
                # expand that list into the index (and subsequent indicies)
                output[i:i + 1] = output[i]
    except IndexError:
        pass
    if return_type == list:
        return output
    return return_type(output)


_ssl_init_done = False


def initialize_ssl_cert_dir():
    """Initialize OpenSSL's CA certificates file.

    Makes it so certificates can be verified.
    """
    global _ssl_init_done
    if _ssl_init_done:
        return
    _ssl_init_done = True

    if not sys.platform.startswith('linux'):
        return
    import os
    import ssl
    dvp = ssl.get_default_verify_paths()
    # from https://golang.org/src/crypto/x509/root_linux.go
    cert_files = [
        "/etc/ssl/certs/ca-certificates.crt",                # Debian/Ubuntu/Gentoo etc.
        "/etc/pki/tls/certs/ca-bundle.crt",                  # Fedora/RHEL 6
        "/etc/ssl/ca-bundle.pem",                            # OpenSUSE
        "/etc/pki/tls/cacert.pem",                           # OpenELEC
        "/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem", # CentOS/RHEL 7
        "/etc/ssl/cert.pem",                                 # Alpine Linux
    ]
    for fn in cert_files:
        if os.path.exists(fn):
            os.environ[dvp.openssl_cafile_env] = fn
            # os.environ[dvp.openssl_capath_env] = os.path.dirname(fn)
            return


def can_set_file_icon():
    '''Can an icon image be associated with a file on this operating system.'''
    from sys import platform
    return platform == 'darwin'


def set_file_icon(path, image):
    '''Assoicate an icon image with a file to be shown by the operating system file browser.'''
    if not can_set_file_icon():
        return

    # Encode image as jpeg.
    import io
    f = io.BytesIO()
    image.save(f, 'JPEG')
    s = f.getvalue()

    from . import _mac_util
    _mac_util.set_file_icon(path, s)

# for backwards compatibility, make string_to_attr importable from this module
from .attributes import string_to_attr

short_words = set(["a", "an", "and", "as", "at", "but", "by", "for", "from", "in", "into", "is", "of",
    "on", "or", "the", "to", "with", "within"])
def titleize(text):
    capped_words = []
    words = text.split()
    while words:
        word = words[0]
        if word[0] in '([{':
            match_char = ')]}'['([{'.index(word[0])]
            if word[-1] == match_char:
                capped_words.append(word[0] + titleize(word[1:-1]) + match_char)
                words = words[1:]
            else:
                for i, matching_word in enumerate(words[1:]):
                    if matching_word[-1] == match_char:
                        capped_words.append(word[0] + titleize(" ".join([word[1:]] + words[1:i+1]
                            + [matching_word[:-1]])) + match_char)
                        words = words[i+2:]
                        break
                else:
                    # no matching closing paren/bracket/brace
                    capped_words.append(word)
                    words = words[1:]
            continue
        if word.lower() != word or (capped_words and word in short_words and len(words) > 1):
            capped_words.append(word)
        else:
            capped_word = ""
            for frag in [x for part in word.split('/') for x in part.split('-')]:
                capped_word += frag.capitalize()
                if len(capped_word) < len(word):
                    capped_word += word[len(capped_word)]
            capped_words.append(capped_word)
        words = words[1:]
    return " ".join(capped_words)

class CustomSortString(str):

    def __new__(cls, str_val,  sort_val=None):
        obj = str.__new__(cls, str_val)
        obj.sort_val = sort_val
        return obj

    def lower(self, *args, **kw):
        return CustomSortString(super().lower(*args, **kw), sort_val=self.sort_val)

    def upper(self, *args, **kw):
        return CustomSortString(super().upper(*args, **kw), sort_val=self.sort_val)

    def __hash__(self):
        return super().__hash__()

    def __lt__(self, other):
        if isinstance(other, CustomSortString):
            if self.sort_val == other.sort_val:
                return super().__lt__(other)
            return self.sort_val < other.sort_val
        return super().__lt__(other)

    def __le__(self, other):
        if isinstance(other, CustomSortString):
            if self.sort_val == other.sort_val:
                return super().__le__(other)
            return self.sort_val <= other.sort_val
        return super().__le__(other)

    def __eq__(self, other):
        if isinstance(other, CustomSortString):
            if self.sort_val == other.sort_val:
                return super().__eq__(other)
            return False
        return super().__eq__(other)

    def __ne__(self, other):
        if isinstance(other, CustomSortString):
            if self.sort_val == other.sort_val:
                return super().__ne__(other)
            return True
        return super().__ne__(other)

    def __gt__(self, other):
        if isinstance(other, CustomSortString):
            if self.sort_val == other.sort_val:
                return super().__gt__(other)
            return self.sort_val > other.sort_val
        return super().__gt__(other)

    def __ge__(self, other):
        if isinstance(other, CustomSortString):
            if self.sort_val == other.sort_val:
                return super().__ge__(other)
            return self.sort_val >= other.sort_val
        return super().__ge__(other)

def round_off(val, significant_digits):
    """Reduce a numerical value to the specified number of significant digits"""
    if val == 0.0:
        return val
    from math import log10, floor
    return round(val, significant_digits - 1 - int(floor(log10(abs(val)))))

def make_link(target, source) -> None:
    """An OS-agnostic way to make a symbolic link that does not require permissions
    on Windows to use."""
    if sys.platform == "win32":
        subprocess.run('mklink /J "%s" "%s"' % (source, target), shell = True)
    else:
        os.symlink(target, source)

def chimerax_binary_directory():
    """Find the ChimeraX.app/Contents/bin directory."""
    using_chimerax = "chimerax" in os.path.realpath(sys.executable).split(os.sep)[-1].lower()
    if using_chimerax or sys.platform != "darwin":
        return os.path.dirname(os.path.realpath(sys.executable))
    # /path/to/ChimeraX.app/Contents/Library/Frameworks/Python.Framework/Versions/3.11/bin/python3.11
    # So we need to trim off everything after 'Contents' and add 'bin'
    return os.sep.join([*os.path.realpath(sys.executable).split(os.sep)[:-7], "bin"])

@contextmanager
def chimerax_bin_dir_first_in_path():
    """Put ChimeraX's bin directory first on the PATH. This is useful to override some library's call to
    a system binary, ensuring the system finds the binary shipped with ChimeraX first."""
    oldpath = os.environ['PATH']
    os.environ['PATH'] = ":".join([chimerax_binary_directory(), oldpath])
    yield
    os.environ['PATH'] = oldpath or ''
