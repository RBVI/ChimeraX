
# vim: set expandtab shiftwidth=4 softtabstop=4 fileencoding=utf-8:

# === UCSF ChimeraX Copyright ===
# Copyright 2023 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

# Make flatpak desktop and metainfo files
# https://www.freedesktop.org/software/appstream/docs/chap-Metadata.html
#

import contextlib
from . import xdg

KEYWORDS = [
    "Protein DataBank",
]


@contextlib.contextmanager
def tag(name, end=None, file=None, **kw):
    args = ' '.join(f'{key}="{value}"' for key, value in kw.items())
    if args:
        args = ' ' + args
    print(f"<{name}{args}>", end=end, file=file)
    yield
    print(f"</{name}>", file=file)


def empty_tag(name, end=None, file=None, **kw):
    args = ' '.join(f'{key}="{value}"' for key, value in kw.items())
    if args:
        args = ' ' + args
    print(f"<{name}{args} />", end=end, file=file)


def make_metainfo(session, info, verbose=False):
    from chimerax.core import buildinfo
    year = buildinfo.date.split('-', 1)[0]
    filename = f"{info.ident}.metainfo.xml"
    if verbose:
        print("generating", filename)
    mime_types = xdg.get_mime_types(session)
    with open(filename, mode='wt', encoding='utf-8') as f:
        print('<?xml version="1.0" encoding="UTF-8"?>', file=f)
        print(f'<!-- Copyright {year} Regents of the University of California -->', file=f)
        with tag("component", type="desktop-application", file=f):
            with tag("id", end='', file=f):
                print(f"{info.ident}", end='', file=f)
            with tag("metadata_license", end='', file=f):
                print("CC0-1.0", end='', file=f)
            with tag("name", end='', file=f):
                print(f"{info.app_author} {info.app_name}", end='', file=f)
            with tag("summary", file=f):
                print("An extensible molecular modeling system, https://www.rbvi.ucsf.edu/chimerax", file=f)
            with tag("icon", type="remote", width=512, height=512, file=f):
                # TODO: put icon on website
                print("https://github.com/RBVI/ChimeraX/raw/develop/src/apps/ChimeraX/ChimeraX-icon512.png", file=f)
            with tag("description", file=f):
                # only tags allowed are: p, ol, ul, li, em, code and no nested lists -->
                print("""<p>
  UCSF ChimeraX (or simply ChimeraX)
  is the next-generation molecular visualization program from the
  Resource for Biocomputing,
  Visualization, and Informatics (RBVI).
  ChimeraX is available free of charge
  for academic, government, nonprofit, and personal use.
  Commercial users, please see
  ChimeraX commercial licensing, https://www.rbvi.ucsf.edu/chimerax/commercial_license.html.
</p>
<p>
  ChimeraX is developed with support from
  National Institutes of Health R01-GM129325,
  Chan Zuckerberg Initiative grant EOSS4-0000000439,
  and the Office of Cyber Infrastructure and Computational Biology,
  National Institute of Allergy and Infectious Diseases.
</p>""", file=f)
            with tag("categories", file=f):
                for category in xdg.CATEGORIES:
                    with tag("category", end='', file=f):
                        print(category, end='', file=f)
            with tag("keywords", file=f):
                for keyword in KEYWORDS:
                    with tag("keyword", end='', file=f):
                        print(keyword, end='', file=f)
            with tag("url", type="homepage", end='', file=f):
                print("https://www.rbvi.ucsf.edu/chimerax/", end='', file=f)
            with tag("url", type="help", end='', file=f):
                print("https://www.rbvi.ucsf.edu/chimerax/docs/contact.html", end='', file=f)
            with tag("launchable", type="desktop-id", end='', file=f):
                print(f"{info.ident}.desktop", end='', file=f)
            with tag("releases", file=f):
                empty_tag("release", version=buildinfo.version, date=buildinfo.date, file=f)
            with tag("provides", file=f):
                with tag("binary", end='', file=f):
                    print("/app/bin/ChimeraX", end='', file=f)
                for mime_type in mime_types:
                    with tag("mediatype", end='', file=f):
                        print(mime_type, end='', file=f)
            with tag("requires", file=f):
                with tag("kernel", version="3.11", compare="ge", end='', file=f):
                    # for Qt 6
                    print("Linux", end='', file=f)
            with tag("recommends", file=f):
                with tag("internet", end='', file=f):
                    print("first-run", end='', file=f)
            with tag("supports", file=f):
                with tag("internet", end='', file=f):
                    print("always", end='', file=f)
            with tag("project_license", end='', file=f):
                print("LicenseRef-proprietary=https://www.rbvi.ucsf.edu/chimerax/docs/license.html", end='', file=f)
            with tag("developer_name", end='', file=f):
                print("UCSF RBVI", end='', file=f)
            with tag("screenshots", file=f):
                with tag("screenshot", type="default", file=f):
                    with tag("caption", end='', file=f):
                        print("ChimeraX main window", end='', file=f)
                    with tag("image", type="source", width=772, height=517, end='', file=f):
                        print("https://www.rbvi.ucsf.edu/chimerax/docs/quickstart/images/chimerax.png", end='', file=f)
                # TODO: provide additional screenshots from features and gallery
            with tag("update_contact", end='', file=f):
                print("https://www.rbvi.ucsf.edu/chimerax/docs/contact.html", end='', file=f)
            empty_tag("content_rating", type='oars-1.1', file=f)


def flatpak_files(session, ident, info=None, verbose=False):
    if not info:
        info = xdg.get_info(session, True, verbose=verbose)
    info.ident = ident
    info.desktop = f"{info.ident}.desktop"
    from chimerax.core.__main__ import localized_app_name
    xdg.make_desktop(session, info, localized_app_name, verbose=verbose)
    make_metainfo(session, info, verbose)
