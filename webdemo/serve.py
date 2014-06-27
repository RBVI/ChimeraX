#!/usr/bin/env python

def run(directory, port, demo=False):
    if directory:
        import os
        os.chdir(directory)
    from BaseHTTPServer import HTTPServer
    from ChimeraHTTPServer import ChimeraHTTPRequestHandler
    server = HTTPServer(('localhost', port), ChimeraHTTPRequestHandler)
    if demo:
        import webbrowser
        webbrowser.open('http://localhost:%d/demo.html' % server.server_port)
    server.serve_forever()

if __name__ == '__main__':
    import sys
    def usage():
        print >> sys.stderr, ("Usage: %s"
                " [-d|--directory dir]"
                " [-p|--port port]"
                " [--demo]") % sys.argv[0]
        raise SystemExit, 2
    import os
    directory = os.path.join(os.path.dirname(__file__), "www")
    port = 8000
    demo = False
    import getopt
    try:
        opts, args = getopt.getopt(sys.argv[1:], "d:p:", ["directory=", "port=", "demo"])
    except getopt.error:
        usage()
    for opt, value in opts:
        if opt in ('-d', '--directory'):
            directory = value
        if opt in ('-p', '--port'):
            port = int(value)
        elif opt == '--demo':
            demo = True
    if args:
        usage()

    run(directory, port, demo=demo)
