#!/bin/env python

# --- UCSF Chimera Copyright ---
# Copyright (c) 2000 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  This notice must be embedded in or
# attached to all copies, including partial copies, of the
# software or any revisions or derivations thereof.
# --- UCSF Chimera Copyright ---

import sys, os, string
if not sys.argv[2:]:
	print('usage:', sys.argv[0], 'Python-src-dir file.py ...')
	raise SystemExit(2)

freezesrc = sys.argv[1] + '/Tools/freeze'
sys.path.insert(0, freezesrc)

import makefreeze

class Module:
	def __init__(self, path):
		self.__path__ = path
		contents = open(path, "r")
		self.__code__ = compile(contents.read() + '\n', arg, 'exec')

#frozen = open("frozen.c", "w")
dict = {}
for arg in sys.argv[2:]:
	base = os.path.basename(arg)
	mod, ext = os.path.splitext(base)
	dict[mod] = Module(arg)
makefreeze.makefreeze('./', dict, debug=1)
