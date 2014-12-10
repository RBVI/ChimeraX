# vim: set expandtab shiftwidth=4 softtabstop=4:
#
# For each filename given on the command line
#    if it is a symbolic link to an absolute location,
#    then replace it with a relative one.
from __future__ import print_function

import os
import sys

for filename in sys.argv[1:]:
    if not os.path.exists(filename):
        print("warning:", filename, "is missing", file=sys.stderr)
        continue
    source = os.path.abspath(filename)
    if not os.path.islink(source):
        #print("warning:", filename, "is not a symbolic link", file=sys.stderr)
        continue
    link = os.readlink(source)
    if link[0] != '/':
        continue
    rellink = os.path.relpath(link, os.path.dirname(source))
    try:
        os.remove(source)
    except OSError:
        print("error: unable to remove", filename, file=sys.stderr)
        continue
    print('symbolic linking', source, 'to', rellink)
    os.symlink(rellink, source)
