# vim: set expandtab shiftwidth=4 softtabstop=4:
#
# Copyright © 2014-2015 Regents of the University of California.
# All Rights Reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  This notice must be embedded in or
# attached to all copies, including partial copies, of the
# software or any revisions or derivations thereof.
#
# Make ChimeraX's Info.plist
#
# Follow the Apple "Runtime Configuration Guidelines" document,
# <http://developer.apple.com/documentation/MacOSX/Conceptual/BPRuntimeConfig/BPRuntimeConfig.html>,
# chapter on "Guidelines for Configuring Applications".
#
# Started this script, just wanting to generate
# Apple Uniform Type Information (UTI) for ChimeraX file types.
# TODO: need domains/homepage each type, may alter registration.
#

import os
import sys
import plistlib
import datetime
from distlib.version import NormalizedVersion as Version
from chimerax.core import configfile

configfile.only_use_defaults = True

app_name = sys.argv[1]

more_info = {
    "Python": {
        "www": "www.python.org",
        "ConformsTo": "public.python-script",
    },
    "VRML": {
        "www": "www.web3d.org",
    },
    "Mol2": {
        "www": "http://www.tripos.com/tripos_resources/fileroot/mol2_format_Dec07.pdf",  # noqa
    },
    "Bild": {
        "www": "cgl.ucsf.edu",
    },
}


def utid(f):
    """convert ChimeraX type to its Apple Universal Type Identifier"""
    # look for a domain the fetch data
    from chimerax.core import io
    for fetch_info in io.fetch._fetchInfo:
        dbname = fetch_info[0]
        homepage = fetch_info[4]
        if f == dbname:
            domain = homepage
            break
    else:
        try:
            domain = more_info[f]['www']
        except KeyError:
            if f.startswith('Chimera'):
                domain = "cgl.ucsf.edu"
            else:
                return None
    from urlparse import urlparse
    if ':' not in domain:
        domain = urlparse("http://%s" % domain)[1]
    else:
        domain = urlparse(domain)[1]
    # TODO: use something on other f.title() because VRML->Vrml
    name = ''.join([a[0].upper() + a[1:] for a in f.split()])
    id = '.'.join(domain.split('.')[::-1]) + '.%s' % name
    return id


def format_name(f):
    category = f.category
    if category == 'Miscellaneous':
        return f.name
    else:
        return '%s %s' % (f.name, category)


def dump_format(f):
    """output Apple Universal Type information for ChimeraX file format"""
    id = utid(f.name)
    if id is None:
        print("skipping", f, file=sys.stderr)
        return
    d = {
        "UTTypeIdentifier": id,
        "UTTypeDescription": format_name(f),
        "UTConformsTo": [more_info[f].get("ConformsTo", "public.data")]
    }
    extensions = f.extensions
    mime_types = f.mime_types
    if extensions or mime_types:
        d2 = d["UTTypeTagSpecification"] = {}
        if extensions:
            d2["public.filename-extension"] = [e[1:] for e in extensions]
        if mime_types:
            d2["public.mime-type"] = mime_types[0]
    return d


# Initialize ChimeraX to get all registered file types
if 'session' in locals() or 'session' in globals():
    formats = session.open_command.open_data_formats  # NOQA
    chimera_types = [f.name for f in formats if f.name.startswith('Chimera')]
else:
    sys.path.insert(0, '')
    from ChimeraX_main import init
    init([app_name, "--safemode", "--nogui", "--exit"])
    formats = []

# create Info.plist

# use today's year as the copyright year
year = datetime.datetime.now().year

# extract chimerax.core version
f = open('../../core/Makefile')
for line in f.readlines():
    if line.startswith('BUNDLE_VERSION'):
        break
else:
    print('error: unable to find chimerax.core version')
    raise SystemExit(1)

version = line.split()[2]
epoch, release, *_ = Version(version).parse(version)
if len(release) == 1:
    release += (0,)
if len(release) < 4:
    release += (0,) * (4 - len(release))

pl = {
    "CFBundleDevelopmentRegion": "English",
    # name of executable in .app/Contents/MacOS/
    "CFBundleExecutable": app_name,
    "NSHumanReadableCopyright": "Copyright \u00A9 %s" % year +
                                " Regents of the University of California."
                                "  All Rights Reserved.",
    "CFBundleIconFile": "chimerax-icon.icns",
    "CFBundleIdentifier": "edu.ucsf.cgl.%s" % app_name,
    "CFBundleInfoDictionaryVersion": "6.0",
    "CFBundleName": "UCSF ChimeraX",
    "CFBundleDisplayName": "UCSF ChimeraX",
    "LSHasLocalizedDisplayName": False,
    "CFBundlePackageType": "APPL",
    "CFBundleShortVersionString": '%s.%s.%s' % tuple(release[0:3]),
    "CFBundleSignature": "Chim",
    "CFBundleVersion": '.'.join(str(i) for i in release),
    # Enable full resolution text on retina displays
    "NSHighResolutionCapable": True,
    "NSAppTransportSecurity": {"NSAllowsArbitraryLoads": True},
    "LSBackgroundOnly": False,
    "NSCameraUsageDescription": "The ChimeraX webcam command needs permission to use the camera.",
    "NSMicrophoneUsageDescription": "The ChimeraX webcam command needs permission to use the microphone.",
}

target = "10.4"
if "MACOSX_DEPLOYMENT_TARGET" in os.environ:
    target = os.getenv("MACOSX_DEPLOYMENT_TARGET")
    pl["LSMinimumSystemVersion"] = "%s.0" % target
useLSItemContent_types = float(target) >= 10.5
# TODO: debug why useLSItemContent_types causes ChimeraX application
# to not work for double-clicking PDB files
useLSItemContent_types = False

pl["CFBundleDocumentTypes"] = []
formats.sort(key=lambda f: f.name)  # get consistent order
for f in formats:
    if useLSItemContent_types:
        id = utid(f.name)
        if not id:
            continue
    else:
        extensions = f.suffixes
        mime_types = f.mime_types
        if not extensions and not mime_types:
            continue
    d = {
        "CFBundleTypeName": format_name(f),
        "CFBundleTypeRole": "Editor",
        # "CFBundleTypeIcon": "???.icns",
    }
    if useLSItemContent_types:
        # do not use LSItemContents_types in
        # 10.4 because it just confuses things
        d["LSItemContentTypes"] = [id]
    else:
        if extensions:
            d["CFBundleTypeExtensions"] = [e[1:] for e in extensions]
        if mime_types:
            d["CFBundleTypeMIMETypes"] = [m for m in mime_types]
    pl["CFBundleDocumentTypes"].append(d)

    if useLSItemContent_types:
        # UTI keys are ignored in Mac OS X v10.4
        # but used in Mac OS X v10.5
        type_info = []
        for f in chimera_types:
            d = dump_format(f)
            if d:
                type_info.append(d)
        if type_info:
            pl["UTExportedTypeDeclarations"] = type_info

        type_info = []
        for f in session.data_formats.formats:  # NOQA
            if f.name in chimera_types:
                continue
            d = dump_format(f)
            if d:
                type_info.append(d)
        if type_info:
            pl["UTImportedTypeDeclarations"] = type_info

with open("Info.plist", "wb") as f:
    plistlib.dump(pl, f)
