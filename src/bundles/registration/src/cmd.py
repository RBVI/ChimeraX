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

from chimerax.core.commands import CmdDesc
from chimerax.core.commands import StringArg, BoolArg, EnumOf, ListOf, NoArg, OpenFileNameArg

ResearchAreas = ["atomic structure analysis",
                 "cryoEM",
                 "light microscopy",
                 "drug design",
                 "teaching",
                 "presentation/publication",
                 "other"]
FundingSources = ["NIH",
                  "NSF",
                  "EMBO",
                  "Wellcome Trust",
                  "other"]
RegistrationURL = "https://www.rbvi.ucsf.edu/chimerax/cgi-bin/chimerax_registration.py"
# RegistrationURL = "https://preview.rbvi.ucsf.edu/chimerax/cgi-bin/chimerax_registration.py"
DiscussionURL = "https://www.rbvi.ucsf.edu/mailman/subscribe/chimerax-users"
AnnouncementsURL = "https://www.rbvi.ucsf.edu/mailman/subscribe/chimerax-announce"
ThankYou = """Thank you for registering your copy of ChimeraX.
By providing the information requested you will
be helping us document the impact this software
is having in the scientific community.  The
information you supplied will only be used
for reporting summary usage statistics; no
individual data will be released."""


def register(session, name, email, organization=None,
             research=None, research_other="",
             funding=None, funding_other="", comment=None,
             join_discussion=False, join_announcements=True):
    from chimerax.core.errors import UserError
    from .nag import check_registration
    expiration = check_registration()
    if expiration is not None:
        session.logger.info("Your copy of Chimera is already registered "
                            "through %s." % expiration.strftime("%x"))
    # Normalize input
    name = name.strip()
    email = email.strip()
    organization = organization.strip() if organization is not None else ""
    research = [u.strip() for u in research] if research is not None else []
    research_other = research_other.strip()
    funding = [f.strip() for f in funding] if funding is not None else []
    funding_other = funding_other.strip()
    comment = comment.strip() if comment is not None else ""
    # Do some error checking
    if not name:
        raise UserError('"Name" field cannot be empty')
    if not email or '@' not in email:
        raise UserError('"E-mail" field cannot be empty or invalid')
    if "other" in research and not research_other:
        raise UserError('"Other research area" field cannot be empty '
                        'when "other" is selected')
    if "other" in funding and not funding_other:
        raise UserError('"Other funding source" field cannot be empty '
                        'when "other" is selected')

    # Get registration from server
    registration = _get_registration(name, email, organization,
                                     research, research_other,
                                     funding, funding_other, comment)
    from .nag import install
    if not install(session, registration):
        # Do not join mailing lists if we cannot install registration data
        return
    session.logger.info(ThankYou, is_html=True)

    # Register for mailing lists
    if join_discussion:
        _subscribe(session, "discussion", DiscussionURL, name, email)
    if join_announcements:
        _subscribe(session, "announcements", AnnouncementsURL, name, email)


def _get_registration(name, email, organization, research, research_other,
                      funding, funding_other, comment):
    from chimerax.core.errors import UserError
    from urllib.parse import urlencode
    from urllib.request import urlopen
    from urllib.error import URLError
    from xml.dom import minidom
    from xml.parsers.expat import ExpatError
    # Required fields
    params = [
        ("action", "Register from ChimeraX"),
        ("name", name),
        ("email", email),
    ]
    # Optional fields
    if organization:
        params.append(("organization", organization))
    for r in research:
        params.append(("research", r))
    if "other" in research:
        params.append(("research_other", research_other))
    for f in funding:
        params.append(("funding", f))
    if "other" in funding:
        params.append(("funding_other", funding_other))
    if comment:
        params.append(("comment", comment))
    try:
        with urlopen(RegistrationURL, urlencode(params).encode()) as f:
            text = f.read()
    except URLError:
        raise UserError("Registration server unavailable.  "
                        "Please try again later.")
    try:
        dom = minidom.parseString(text)
    except ExpatError:
        raise UserError("Registration server error.  Please try again later.")
    registration = _get_tag_text(dom, "registration")
    if not registration:
        error = _get_tag_text(dom, "error")
        if not error:
            raise UserError("Registration failed.  Please try again later.")
        else:
            raise UserError(error)
    return registration


def _get_tag_text(dom, tag_name):
    text = []
    for e in dom.getElementsByTagName(tag_name):
        text.append(_get_text(e))
    return ''.join(text)


def _get_text(e):
    text = []
    for node in e.childNodes:
        if node.nodeType == node.TEXT_NODE:
            text.append(node.data)
    return ''.join(text)


def _subscribe(session, label, url, name, email):
    from urllib.parse import urlencode
    from urllib.request import urlopen
    from urllib.error import URLError
    params = {
        "fullname": name,
        "email": email,
    }
    try:
        with urlopen(url, urlencode(params).encode()) as f:
            text = f.read()
            del text
        session.logger.info("%s is subscribed to the ChimeraX %s list" %
                            (email, label))
    except URLError as e:
        session.logger.warning("Failed to subscribed %s to the ChimeraX "
                               "%s list: %s" % (email, label, str(e)))


register_desc = CmdDesc(keyword=[("name", StringArg),
                                 ("email", StringArg),
                                 ("organization", StringArg),
                                 ("research", ListOf(EnumOf(ResearchAreas))),
                                 ("research_other", StringArg),
                                 ("funding", ListOf(EnumOf(FundingSources))),
                                 ("funding_other", StringArg),
                                 ("comment", StringArg),
                                 ("join_discussion", BoolArg),
                                 ("join_announcements", BoolArg)],
                        required_arguments=["name", "email"])


def registration_status(session, verbose=False):
    from .nag import report_status
    report_status(session.logger, verbose)


registration_status_desc = CmdDesc(keyword=[("verbose", NoArg)])


def registration_file(session, filename=None):
    if filename is None:
        from .nag import _registration_file
        session.logger.info("Registration file is %s" % _registration_file())
    else:
        try:
            with open(filename) as f:
                data = f.read()
            try:
                start_index = data.index("<pre>\n") + 6
                end_index = data.index("</pre>\n", start_index)
                reg_data = data[start_index:end_index]
            except ValueError:
                raise IOError("no registration data found")
        except IOError as e:
            session.logger.error("%s: %s" % (filename, str(e)))
        else:
            from .nag import install
            if install(session, reg_data):
                session.logger.info(ThankYou, is_html=True)


registration_file_desc = CmdDesc(optional=[("filename", OpenFileNameArg)])
