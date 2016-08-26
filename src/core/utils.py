# vim: set expandtab shiftwidth=4 softtabstop=4:

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

"""
utils: Generically useful stuff that doesn't fit elsewhere
==========================================================
"""


# from Mike C. Fletcher's BasicTypes library
# via http://rightfootin.blogspot.com/2006/09/more-on-python-flatten.html
# Except called flattened, like sorted, since it is nondestructive
def flattened(input, return_types=(list, tuple, set)):
    """Return new flattened version of input

    Parameters
    ----------
    input : a sequence instance (list, tuple, or set)

    Returns
    -------
    A sequence of the same type as the input.
    """
    return_type = type(input)
    output = list(input)
    i = 0
    while i < len(output):
        while isinstance(output[i], return_types):
            if not output[i]:
                output.pop(i)
                i -= 1
                break
            else:
                output[i:i + 1] = output[i]
        i += 1
    if return_type == list:
        return output
    return return_type(output)

def initialize_ssl_cert_dir():
    """Initialize OpenSSL's CA certificates file.

    Makes it so certificates can be verified.
    """
    global _ssl_init_done
    if _ssl_init_done:
        return
    _ssl_init_done = True

    import sys
    if not sys.platform.startswith('linux'):
        return
    import os
    import ssl
    dvp = ssl.get_default_verify_paths()
    # from https://golang.org/src/crypto/x509/root_linux.go
    cert_files = [
        "/etc/ssl/certs/ca-certificates.crt",  # Debian/Ubuntu/Gentoo etc.
        "/etc/pki/tls/certs/ca-bundle.crt",    # Fedora/RHEL
        "/etc/ssl/ca-bundle.pem",              # OpenSUSE
        "/etc/pki/tls/cacert.pem",             # OpenELEC
    ]
    for fn in cert_files:
        if os.path.exists(fn):
            os.environ[dvp.openssl_cafile_env] = fn
            # os.environ[dvp.openssl_capath_env] = os.path.dirname(fn)
            return
_ssl_init_done = False


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
