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

# Patch pypiserver to serve up metadata in addition to packages
from pypiserver._app import app

@app.route("/metadata")
def metadata():
    from pypiserver.bottle import static_file
    from pypiserver._app import packages, config
    resp = static_file("METADATA.json", root=packages.root, mimetype="application/json")
    if config.cache_control:
        resp.set_header("Cache-Control", "public, max-age=%s" % config.cache_control)
    return resp

from pypiserver import __main__
__main__.main()
