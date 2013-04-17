"""Chimera-savvy HTTP Server.

This module builds on CGIHTTPServer by replacing do_POST that
intercepts requests starting with /chimera-app and running them
within Chimera instead of as CGI

SECURITY WARNING: DON'T USE THIS CODE UNLESS YOU ARE INSIDE A FIREWALL
-- it may execute arbitrary Python code or external programs.

Note that status code 200 is sent prior to execution of a CGI script, so
scripts cannot send other status codes such as 302 (redirect).
"""

__version__ = "0.4"

__all__ = ["CGIHTTPRequestHandler"]

import os
import sys
import urllib
import BaseHTTPServer
import CGIHTTPServer
import select

# for chimera2 package
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

class ChimeraHTTPRequestHandler(CGIHTTPServer.CGIHTTPRequestHandler):

    """Complete HTTP server with GET, HEAD and POST commands.

    The POST command is *only* implemented for Chimera and CGI scripts.

    """

    def do_POST(self):
        """Serve a POST request."""

        if self.is_chimera():
            self.run_chimera()
        else:
            CGIHTTPServer.CGIHTTPRequestHandler.do_POST(self)

    def do_GET(self):
        """Serve a GET request."""

        if self.is_chimera():
            self.run_chimera()
        else:
            CGIHTTPServer.CGIHTTPRequestHandler.do_GET(self)

    def do_HEAD(self):
        """Serve a HEAD request."""

        if self.is_chimera():
            self.run_chimera()
        else:
            CGIHTTPServer.CGIHTTPRequestHandler.do_GET(self)


    def is_chimera(self):
        """Test whether self.path corresponds to a Chimera script."""
	chimera_app = "/chimera-app/"
	if not self.path.startswith(chimera_app):
	    return False
	self.cgi_info = (chimera_app[1:-1], self.path[len(chimera_app):])
	return True

    def run_chimera(self):
        """Execute a CGI script."""
        path = self.path
        app, rest = self.cgi_info

	import shlex
	try:
	    chimera_cmd = shlex.split(urllib.unquote_plus(rest))
	except ValueError, e:
	    self.send_error(400, "Syntax Error: %s" % e)
	    return

        # Reference: http://hoohoo.ncsa.uiuc.edu/cgi/env.html
        # XXX Much of the following could be prepared ahead of time!
        env = {}
        env['SERVER_SOFTWARE'] = self.version_string()
        env['SERVER_NAME'] = self.server.server_name
        env['GATEWAY_INTERFACE'] = 'CGI/1.1'
        env['SERVER_PROTOCOL'] = self.protocol_version
        env['SERVER_PORT'] = str(self.server.server_port)
        env['REQUEST_METHOD'] = self.command
        host = self.address_string()
        if host != self.client_address[0]:
            env['REMOTE_HOST'] = host
        env['REMOTE_ADDR'] = self.client_address[0]
        authorization = self.headers.getheader("authorization")
        if authorization:
            authorization = authorization.split()
            if len(authorization) == 2:
                import base64, binascii
                env['AUTH_TYPE'] = authorization[0]
                if authorization[0].lower() == "basic":
                    try:
                        authorization = base64.decodestring(authorization[1])
                    except binascii.Error:
                        pass
                    else:
                        authorization = authorization.split(':')
                        if len(authorization) == 2:
                            env['REMOTE_USER'] = authorization[0]
        # XXX REMOTE_IDENT
        if self.headers.typeheader is None:
            env['CONTENT_TYPE'] = self.headers.type
        else:
            env['CONTENT_TYPE'] = self.headers.typeheader
        length = self.headers.getheader('content-length')
        if length:
            env['CONTENT_LENGTH'] = length
        referer = self.headers.getheader('referer')
        if referer:
            env['HTTP_REFERER'] = referer
        accept = []
        for line in self.headers.getallmatchingheaders('accept'):
            if line[:1] in "\t\n\r ":
                accept.append(line.strip())
            else:
                accept = accept + line[7:].split(',')
        env['HTTP_ACCEPT'] = ','.join(accept)
        ua = self.headers.getheader('user-agent')
        if ua:
            env['HTTP_USER_AGENT'] = ua
        co = filter(None, self.headers.getheaders('cookie'))
        if co:
            env['HTTP_COOKIE'] = ', '.join(co)
        # XXX Other HTTP_* headers
        # Since we're setting the env in the parent, provide empty
        # values to override previously set values
        for k in ('QUERY_STRING', 'REMOTE_HOST', 'CONTENT_LENGTH',
                  'HTTP_USER_AGENT', 'HTTP_COOKIE', 'HTTP_REFERER'):
            env.setdefault(k, "")
        env.update(os.environ)

	#self.send_response(202, "Script output follows")

        saveStdout = sys.stdout
        sys.stdout = self.wfile
        saveStdin = sys.stdin
        sys.stdin = self.rfile
        saveArgv = sys.argv
        sys.argv = chimera_cmd
        saveEnviron = os.environ
        os.environ = env
        try:
            try:
		from chimera2 import webcmd
		webcmd.webcmd(self)
            finally:
                sys.stdout = saveStdout
                sys.stdin = saveStdin
                sys.argv = saveArgv
                os.environ = saveEnviron
                self.wfile.flush()
        except:
            import traceback, cgi
            text = traceback.format_exc()
	    self.send_error(500, cgi.escape(text))
            self.log_error("Script %s raised exception" % sys.argv[0])
        else:
            self.log_message("Script %s exited OK" % sys.argv[0])


if __name__ == '__main__':
    from CGIHTTPServer import test
    test()
