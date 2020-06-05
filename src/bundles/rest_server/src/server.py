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

from http.server import BaseHTTPRequestHandler
from chimerax.core.tasks import Task
from chimerax.core.logger import PlainTextLog

class RESTServer(Task):
    """Listen for HTTP/REST requests, execute them and return output."""

    def __init__(self, *args, **kw):
        import threading
        self.httpd = None
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
    def server_address(self):
        return self.httpd.server_address

    def run(self, port, use_ssl):
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
            self.httpd.socket = context.wrap_socket(self.httpd.socket,
                                                    server_side=True)
        self.run_increment()    # To match decrement in terminate()
        host, port = self.httpd.server_address
        msg = ("REST server started on host %s port %d" % (host, port))
        self.session.ui.thread_safe(print, msg, file=sys.__stdout__, flush=True)
        msg += ('\nVisit %s://%s:%d/cmdline.html for CLI interface' %
               (proto, host, port))
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


class RESTHandler(BaseHTTPRequestHandler):
    """Process one REST request."""

    ContentTypes = [
        (".html", "text/html"),
        (".png", "image/png"),
        (".ico", "image/png"),
    ]

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
                fn = os.path.join(os.path.dirname(__file__),
                                  "static", r.path[1:])
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

    def log_request(self, code='-', size='-'):
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
        def f(args=args, session=session, q=q):
            logger = session.logger
            # rest_log.log_summary gets called at the end
            # of the "with" statement
            with StringPlainTextLog(logger) as rest_log:
                from chimerax.core.commands import run
                try:
                    commands = args["command"]
                except KeyError:
                    logger.error("\"command\" parameter missing")
                else:
                    try:
                        for cmd in commands:
                            if isinstance(cmd, bytes):
                                cmd = cmd.decode('utf-8')
                            run(session, cmd, log=False)
                    except NotABug as e:
                        logger.info(str(e))
                q.put(rest_log.getvalue())
        session.ui.thread_safe(f)
        data = bytes(q.get(), "utf-8")
        self._header(200, "text/plain", len(data))
        self.wfile.write(data)
