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
