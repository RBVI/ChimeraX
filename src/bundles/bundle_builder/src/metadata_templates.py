# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# This software is provided pursuant to the ChimeraX license agreement, which
# covers academic and commercial uses. For more information, see
# <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This file is part of the ChimeraX library. You can also redistribute and/or
# modify it under the GNU Lesser General Public License version 2.1 as
# published by the Free Software Foundation. For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# This file is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
# must be embedded in or attached to all copies, including partial copies, of
# the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===
metadata_preamble = """Framework :: ChimeraX
Intended Audience :: Science/Research
Programming Language :: Python :: 3
Topic :: Scientific/Engineering :: Visualization
Topic :: Scientific/Engineering :: Chemistry
Topic :: Scientific/Engineering :: Bio-Informatics"""

pure_wheel_platforms = """Environment :: MacOS X :: Aqua
Environment :: Win32 (MS Windows)
Environment :: X11 Applications
Operating System :: MacOS :: MacOS X
Operating System :: Microsoft :: Windows :: Windows 10
Operating System :: POSIX :: Linux"""

wheel_file_template = """Wheel-Version: 1.0
Generator: ChimeraX-BundleBuilder ({})
Root-Is-Purelib: {}
Tag: {}"""

metadata_header_template = """Metadata-Version: 2.1
Name: {}
Version: {}"""

def metadata_header(*, name, version, summary = None, homepage = None, author = None, email = None):
    str_ = metadata_header_template.format(name, version)
    if summary:
        str_ = "\n".join([str_, "Summary: {}".format(summary)])
    if homepage:
        str_ = "\n".join([str_, "Home-page: {}".format(homepage)])
    if author:
        str_ = "\n".join([str_, "Author: {}".format(author)])
    if email:
        str_ = "\n".join([str_, "Author-email: {}".format(email)])
    return str_
