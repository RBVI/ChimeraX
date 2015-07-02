#!/bin/env python
# vi: set expandtab shiftwidth=4 softtabstop=4:
#
# Copyright © 2014-2015 Regents of the University of California.
# All Rights Reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  This notice must be embedded in or
# attached to all copies, including partial copies, of the
# software or any revisions or derivations thereof.
#
# Make chimera2's Info.plist
#
# Follow the Apple "Runtime Configuration Guidelines" document,
# <http://developer.apple.com/documentation/MacOSX/Conceptual/BPRuntimeConfig/BPRuntimeConfig.html>,
# chapter on "Guidelines for Configuring Applications".
#
# Started this script, just wanting to generate
# Apple Uniform Type Information (UTI) for chimera file types.
# TODO: need domains/homepage each type, may alter registration.
#

from __future__ import print_function

import os
import sys
import plistlib
from chimera.core import io, session, logger, configfile, core_settings

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
    """convert chimera type to its Apple Universal Type Identifier"""
    # look for a domain the fetch data
    for fetchInfo in io.fetch._fetchInfo:
        dbname = fetchInfo[0]
        homepage = fetchInfo[4]
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
    category = io.category(f)
    if category == 'Miscellaneous':
        return f
    else:
        return '%s %s' % (f, category)


def dump_format(f):
    """output Apple Universal Type information for chimera file format"""
    id = utid(f)
    if id is None:
        print("skipping", f, file=sys.stderr)
        return
    d = {
        "UTTypeIdentifier": id,
        "UTTypeDescription": format_name(f),
        "UTConformsTo": [more_info[f].get("ConformsTo", "public.data")]
    }
    extensions = io.extensions(f)
    mime_types = io.mime_types(f)
    if extensions or mime_types:
        d2 = d["UTTypeTagSpecification"] = {}
        if extensions:
            d2["public.filename-extension"] = [e[1:] for e in extensions]
        if mime_types:
            d2["public.mime-type"] = mime_types[0]
    return d

sess = session.Session()
sess.app_name = "unknown"
sess.debug = False
sess.logger = logger.Logger(sess)
core_settings.init(sess)
session.common_startup(sess)

chimera_types = [f for f in io.formats() if f.startswith('Chimera')]

# create Info.plist
# TODO:
# from chimera.core.version import version, releaseNum
# year = version.split()[5].split('/')[0]
# release = releaseNum[:]
year = 2015
release = [2, 0]
if len(release) < 4:
    release[-1:-1] = [0] * (4 - len(release))

pl = {
    "CFBundleDevelopmentRegion": "English",
    # name of executable in .app/Contents/MacOS/
    "CFBundleExecutable": app_name,
    "NSHumanReadableCopyright": "Copyright \u00A9 %s" % year +
                                " Regents of the University of California."
                                "  All Rights Reserved.",
    "CFBundleIconFile": "chimera-icon.icns",
    "CFBundleIdentifier": "edu.ucsf.cgl.%s" % app_name,
    "CFBundleInfoDictionaryVersion": "6.0",
    "CFBundleName": "UCSF Chimera2",
    "CFBundleDisplayName": "UCSF Chimera2",
    "LSHasLocalizedDisplayName": False,
    "CFBundlePackageType": "APPL",
    "CFBundleShortVersionString": '%s.%s.%s' % tuple(release[0:3]),
    "CFBundleSignature": "Chim",
    "CFBundleVersion": '.'.join(str(i) for i in release),
    # Enable full resolution text on retina displays
    "NSHighResolutionCapable": True,
    "LSBackgroundOnly": False,
}

target = "10.4"
if "MACOSX_DEPLOYMENT_TARGET" in os.environ:
    target = os.getenv("MACOSX_DEPLOYMENT_TARGET")
    pl["LSMinimumSystemVersion"] = "%s.0" % target
useLSItemContent_types = float(target) >= 10.5
# TODO: debug why useLSItemContent_types causes chimera application
# to not work for double-clicking PDB files
useLSItemContent_types = False

pl["CFBundleDocumentTypes"] = []
for f in io.formats():
    if useLSItemContent_types:
        id = utid(f)
        if not id:
            continue
    else:
        extensions = io.extensions(f)
        mime_types = io.mime_types(f)
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
        for f in io.formats():
            if f in chimera_types:
                continue
            d = dump_format(f)
            if d:
                type_info.append(d)
        if type_info:
            pl["UTImportedTypeDeclarations"] = type_info

plistlib.writePlist(pl, 'Info.plist')
