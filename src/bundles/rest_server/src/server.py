# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This particular file is part of the ChimeraX library. You can also
# redistribute and/or modify it under the terms of the GNU Lesser General
# Public License version 2.1 as published by the Free Software Foundation.
# For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
# EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
# LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
# VERSION 2.1
#
# This notice must be embedded in or attached to all copies, including partial
# copies, of the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

from http.server import BaseHTTPRequestHandler
from chimerax.core.tasks import Task
from chimerax.core.logger import PlainTextLog, StringPlainTextLog


class RESTServer(Task):
    """Listen for HTTP/REST requests, execute them and return output."""

    def __init__(self, *args, **kw):
        import threading

        self.httpd = None
        self.log = kw.pop("log", False)
        self.run_count = 0
        self.run_lock = threading.Lock()
        super().__init__(*args, **kw)

    SESSION_SAVE = False

    def take_snapshot(self, session, flags):
        # For now, do not save anything in session.
        # Should save port and auto-restart on session restore.
        return None

    @classmethod
    def restore_snapshot(cls, session, data):
        # For now do nothing.  Should restart on port (in data)
        pass

    @property
    def log(self):
        return self._log

    @log.setter
    def log(self, value):
        self._log = value
        RESTHandler.log = value

    @property
    def server_address(self):
        return self.httpd.server_address

    def run(self, port, use_ssl, json):
        from http.server import HTTPServer
        import sys

        if port is None:
            # Defaults to any available port
            port = 0
        if use_ssl is None:
            # Defaults to cleartext
            use_ssl = False
        self.httpd = HTTPServer(("localhost", port), RESTHandler)
        self.httpd.chimerax_restserver = self
        self.json = json
        if not use_ssl:
            proto = "http"
        else:
            proto = "https"
            try:
                import os.path, ssl
            except ImportError:
                from chimerax.core.errors import LimitationError

                raise LimitationError("SSL is not supported")
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            cert = os.path.join(os.path.dirname(__file__), "server.pem")
            context.load_cert_chain(cert)
            # self.httpd.socket = ssl.wrap_socket(self.httpd.socket,
            #                                     certfile=cert)
            self.httpd.socket = context.wrap_socket(self.httpd.socket, server_side=True)
        self.run_increment()  # To match decrement in terminate()
        host, port = self.httpd.server_address
        msg = "REST server started on host %s port %d" % (host, port)
        self.session.ui.thread_safe(print, msg, file=sys.__stdout__, flush=True)
        msg += "\nVisit %s://%s:%d/cmdline.html for CLI interface" % (proto, host, port)
        self.session.ui.thread_safe(self.session.logger.info, msg)
        self.httpd.serve_forever()

    def run_increment(self):
        with self.run_lock:
            if self.httpd is None:
                return False
            else:
                self.run_count += 1
                return True

    def run_decrement(self):
        with self.run_lock:
            self.run_count -= 1
            if self.run_count == 0:
                if self.httpd is not None:
                    self.httpd.shutdown()
                    self.httpd = None
                super().terminate()

    def terminate(self):
        self.run_decrement()

    def __str__(self):
        return "REST Server, ID %s" % self.id


class RESTHandler(BaseHTTPRequestHandler):
    """Process one REST request."""

    ContentTypes = [
        (".html", "text/html"),
        (".png", "image/png"),
        (".ico", "image/png"),
    ]

    # Whether to log to the ChimeraX log as well as whatever client is being
    # used
    log = False

    def do_GET(self):
        if not self.server.chimerax_restserver.run_increment():
            return
        try:
            from urllib.parse import urlparse, parse_qs

            r = urlparse(self.path)
            if r.path == "/run":
                # Execute a command
                args = parse_qs(r.query)
                if self.command == "POST":
                    for k, vl in self._parse_post().items():
                        try:
                            al = args[k]
                        except KeyError:
                            args[k] = vl
                        else:
                            al.extend(vl)
                self._run(args)
            else:
                # Serve up some static files for testing
                import os.path

                fn = os.path.join(os.path.dirname(__file__), "static", r.path[1:])
                try:
                    with open(fn, "rb") as f:
                        data = f.read()
                    for suffix, ctype in self.ContentTypes:
                        if r.path.endswith(suffix):
                            break
                    else:
                        ctype = "text/plain"
                    self._header(200, ctype, len(data))
                    self.wfile.write(data)
                except IOError:
                    self.send_error(404)
        finally:
            self.server.chimerax_restserver.run_decrement()

    do_POST = do_GET

    def log_request(self, code="-", size="-"):
        pass

    def _parse_post(self):
        ctype = self.headers.get("content-type")
        if not ctype:
            return {}
        from . import cgi

        ctype, pdict = cgi.parse_header(ctype)
        if ctype == "multipart/form-data":
            pdict["boundary"] = pdict["boundary"].encode("utf-8")
            parts = cgi.parse_multipart(self.rfile, pdict)
            return parts
        elif ctype == "application/x-www-form-urlencoded":
            from urllib.parse import parse_qs

            clength = int(self.headers.get("content-length"))
            fields = parse_qs(self.rfile.read(clength), True)
            return fields
        else:
            return {}

    def _header(self, response, content_type, length=None):
        self.send_response(response)
        self.send_header("Content-Type", content_type)
        if length is not None:
            self.send_header("Content-Length", str(length))
        self.end_headers()

    def _run(self, args):
        from queue import Queue
        from chimerax.core.errors import NotABug
        from chimerax.core.logger import StringPlainTextLog

        session = self.server.chimerax_restserver.session
        q = Queue()

        json = self.server.chimerax_restserver.json
        def f(args=args, session=session, q=q, json=json):
            logger = session.logger
            # rest_log.log_summary gets called at the end
            # of the "with" statement
            log_class = ByLevelPlainTextLog if json else StringPlainTextLog
            log_class.propagate_to_chimerax = RESTHandler.log
            with log_class(logger) as rest_log:
                from chimerax.core.commands import run

                error_info = None
                try:
                    commands = args["command"]
                except KeyError:
                    logger.error('"command" parameter missing')
                    ret_val = []
                else:
                    try:
                        for cmd in commands:
                            if isinstance(cmd, bytes):
                                cmd = cmd.decode("utf-8")
                            ret_val = run(
                                session,
                                cmd,
                                log=RESTHandler.log,
                                return_json=json,
                                return_list=True,
                            )
                    except NotABug as e:
                        if json:
                            ret_val = []
                            error_info = e
                        logger.info(str(e))
                    except Exception as e:
                        if json:
                            ret_val = []
                            error_info = e
                        else:
                            raise
                # if json, compose Python and JSON return values into a JSON string,
                # along with log messages broken down by logging level
                if json:
                    from chimerax.core.commands import JSONResult
                    from json import JSONEncoder

                    json_vals = []
                    python_vals = []
                    for val in ret_val:
                        if isinstance(val, JSONResult):
                            json_vals.append(val.json_value)
                            python_vals.append(val.python_value)
                        else:
                            json_vals.append(None)
                            python_vals.append(val)
                    response = {}
                    response["json values"] = json_vals
                    response["python values"] = make_json_friendly(python_vals)
                    response["log messages"] = rest_log.getvalue()
                    if error_info is None:
                        response["error"] = None
                    else:
                        response["error"] = {
                            "type": error_info.__class__.__name__,
                            "message": str(error_info),
                        }
                    q.put(JSONEncoder().encode(response))
                else:
                    q.put(rest_log.getvalue())

        session.ui.thread_safe(f)
        data = bytes(q.get(), "utf-8")
        content_type = "application/json" if json else "text/plain"
        self._header(200, content_type, len(data))
        self.wfile.write(data)

from chimerax.atomic import Structure, AtomicStructure, Chain, Atom, Bond, Residue
try:
    from chimerax.map import Volume
    _stringables = (Structure, AtomicStructure, Chain, Atom, Bond, Residue, Volume)
except ImportError:
    # map bundle may not be installed
    _stringables = (Structure, AtomicStructure, Chain, Atom, Bond, Residue)

def _objects_to_string(objects):
    """Convert Objects instance to a human-readable string summary"""
    from chimerax.core.objects import Objects
    if not isinstance(objects, Objects):
        return str(objects)
    
    # Build summary similar to report_selection
    lines = []
    ac = objects.num_atoms
    bc = objects.num_bonds
    pbc = objects.num_pseudobonds
    rc = objects.num_residues
    mc = len(objects.models)
    
    if mc == 0 and ac == 0 and bc == 0 and pbc == 0:
        return "Nothing"
    
    if ac != 0:
        plural = ('s' if ac > 1 else '')
        lines.append('%d atom%s' % (ac, plural))
    if bc != 0:
        plural = ('s' if bc > 1 else '')
        lines.append('%d bond%s' % (bc, plural))
    if pbc != 0:
        plural = ('s' if pbc > 1 else '')
        lines.append('%d pseudobond%s' % (pbc, plural))
    if rc != 0:
        plural = ('s' if rc > 1 else '')
        lines.append('%d residue%s' % (rc, plural))
    if mc != 0:
        plural = ('s' if mc > 1 else '')
        lines.append('%d model%s' % (mc, plural))
    
    return ', '.join(lines)

def make_json_friendly(val):
    # Check for basic JSON-serializable types first (fast path)
    if val is None or isinstance(val, (bool, int, float, str)):
        return val
    
    # Check for Objects first (before _stringables)
    from chimerax.core.objects import Objects
    if isinstance(val, Objects):
        return _objects_to_string(val)
    
    # Check for Fit objects from map_fit bundle
    try:
        from chimerax.map_fit.search import Fit
        if isinstance(val, Fit):
            return val.fit_message()
    except (ImportError, AttributeError):
        # map_fit bundle may not be installed or Fit class not available
        pass
    
    # Check for known stringable types
    if isinstance(val, _stringables):
        return str(val)
    
    # Recursively handle collections
    if isinstance(val, dict):
        return { make_json_friendly(k): make_json_friendly(v) for k, v in val.items() }
    if isinstance(val, list):
        return [make_json_friendly(v) for v in val]
    if isinstance(val, tuple):
        return tuple(make_json_friendly(v) for v in val)
    
    # Generic fallback for any ChimeraX object: try __str__
    # This handles ToolInstance objects (like Isolde) and other ChimeraX classes
    try:
        # Check if it's a ChimeraX object (has a session attribute)
        if hasattr(val, 'session') or hasattr(val, '__module__') and 'chimerax' in val.__module__:
            return str(val)
    except:
        pass
    
    # Last resort: return the value as-is and let JSON encoder handle it or fail
    return val

class ByLevelPlainTextLog(StringPlainTextLog):
    propagate_to_chimerax = False

    def __init__(self, logger):
        super().__init__(logger)
        self._msgs = {descript: [] for descript in self.LEVEL_DESCRIPTS}

    def log(self, level, msg):
        self._msgs[self.LEVEL_DESCRIPTS[level]].append(msg)
        return not ByLevelPlainTextLog.propagate_to_chimerax

    def getvalue(self):
        return self._msgs
