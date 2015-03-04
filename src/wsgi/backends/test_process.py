#!/usr/bin/python
# vi: set expandtab shiftwidth=4 softtabstop=4:
from base import register

@register(__name__.split('.')[-1])
def backend(sessionDir, sessionName, toChild, fromChild):
	import os, os.path
	os.dup2(toChild[0].fileno(), 0)
	os.dup2(fromChild[1].fileno(), 1)
	toChild[0].close()
	toChild[1].close()
	fromChild[0].close()
	fromChild[1].close()
	os.execl(__file__, os.path.basename(__file__), sessionDir, sessionName)

def main():
	import sys
	sessionDir = sys.argv[1]
	sessionName = sys.argv[2]
	from test_inline import Server
	from _multiprocessing import Connection
	Server(sessionDir, sessionName, Connection(0), Connection(1)).run()
	f.close()

if __name__ == "__main__":
	main()
