# vim: set expandtab ts=4 sw=4:

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

def post_multipart(host, selector, fields, ssl=False, **kw):
    """
Post fields and files to an http host as multipart/form-data.
fields is a sequence of (name, filename, value) elements for form fields.
If filename is None, the field is treated as a regular field;
otherwise, the field is uploaded as a file.
Return the server's response page.
"""
    return post_multipart_formdata(host, selector, fields, ssl, **kw)[3]

def post_multipart_formdata(host, url, fields, ssl=False, *, accept_type=None, timeout=None):
    content_type, body = encode_multipart_formdata(fields)
    from urllib import request
    proxies = request.getproxies_environment()
    try:
        realhost = proxies["http"]
    except KeyError:
        realhost = host
    from http.client import HTTPConnection, HTTPSConnection
    if ssl:
        h = HTTPSConnection(realhost, timeout=timeout)
    else:
        h = HTTPConnection(realhost, timeout=timeout)
    headers = {'Content-type': content_type}
    if accept_type is not None:
        headers['Accept'] = accept_type
    h.request('POST', url, body=body, headers=headers)
    r = h.getresponse()
    return r.status, r.msg, r.getheaders(), r.read()

def encode_multipart_formdata(fields):
    """
fields is a sequence of (name, filename, value) elements for data
to be uploaded as files.  If filename is None, the field is not
given a filename.
Return (content_type, body) ready for httplib.HTTP instance
"""
    BOUNDARY = '---------------------------473995594142710163552326102'
    L = []
    for (key, filename, value) in fields:
        L.append('--' + BOUNDARY)
        if filename is None:
            L.append('Content-Disposition: form-data; name="%s"' % key)
            L.append('Content-Type: text/plain; charset=UTF-8')
        else:
            L.append('Content-Disposition: form-data; name="%s"; filename="%s"' % (key, filename))
            L.append('Content-Type: %s' % get_content_type(filename))
        L.append('')
        L.append(value)
    L.append('--' + BOUNDARY + '--')
    L.append('')
    blines = [(bytes(line, 'utf-8') if isinstance(line, str) else line) for line in L]
    body = b'\r\n'.join(blines)
    content_type = 'multipart/form-data; boundary=%s' % BOUNDARY
    return content_type, body

# on Windows, guessing the mime type involves consulting the Registry, which is not thread safe...
import sys, threading
_lock = None if sys.platform != "win32" else threading.Lock()

def get_content_type(filename):
    import mimetypes
    if _lock is None:
        return mimetypes.guess_type(filename)[0] or 'application/octet-stream'
    with _lock:
        return mimetypes.guess_type(filename)[0] or 'application/octet-stream'
