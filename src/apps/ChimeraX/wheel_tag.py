"""
Script to generate the wheel compatibility tag for the invoking
version of Python.  By default, the generated tag is for a
platform-specific wheel; use -p for a pure-Python wheel.

Output is simply the dash-separated tag, e.g.,
cp36-cp36m-win_amd64
"""

import sys, getopt
pure = False
debug = False
malloc = False
opts, args = getopt.getopt(sys.argv[1:], "p")
for opt, val in opts:
	if opt == "-p":
		pure = True

# Code below is taken from wheel==0.29 bdist_wheel.py
from wheel.pep425tags import get_impl_ver
impl_ver = get_impl_ver()
if pure:
	impl = "py" + impl_ver
	abi = "none"
	platform = "any"
else:
	from wheel.pep425tags import get_abbr_impl, get_abi_tag
	impl = get_abbr_impl() + impl_ver
	# get_abi_tag generates warning messages
	import warnings
	warnings.simplefilter("ignore", RuntimeWarning)
	abi = get_abi_tag()
	from distutils.util import get_platform
	platform = get_platform()

def fix_name(name):
	return name.replace('-', '_').replace('.', '_')
print("%s-%s-%s" % (fix_name(impl), fix_name(abi), fix_name(platform)))
