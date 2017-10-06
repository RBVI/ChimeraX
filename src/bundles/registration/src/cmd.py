# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

from chimerax.core.commands import CmdDesc
from chimerax.core.commands import StringArg, BoolArg, FloatArg, IntArg, EnumOf

OrganizationTypes = ["educational",
                     "non-profit",
                     "commercial",
                     "personal"]
UsageTypes = ["research",
              "teaching",
              "presentation",
              "personal"]
#RegistrationURL = "https://www.rbvi.ucsf.edu/chimerax/cgi-bin/chimerax_registration.py"
RegistrationURL = "https://preview.rbvi.ucsf.edu/chimerax/cgi-bin/chimerax_registration.py"
DiscussionURL = "https://www.rbvi.ucsf.edu/mailman/subscribe/chimerax-users"
AnnouncementsURL = "https://www.rbvi.ucsf.edu/mailman/subscribe/chimerax-announce"
ThankYou = """Thank you for registering your copy of ChimeraX.
By providing the information requested you will
be helping us document the impact this software
is having in the scientific community.  The
information you supplied will only be used
for reporting summary usage statistics; no
individual data will be released."""


def register(session, name, email, organization, type, usage, nih_funded,
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
    organization = organization.strip()
    type = type.strip().casefold()
    usage = usage.strip().casefold()
    nih_funded = "yes" if nih_funded else ""
    # Do some error checking
    if not name:
        raise UserError('"Name" field cannot be empty')
    if not email or '@' not in email:
        raise UserError('"E-mail" field cannot be empty or invalid')

    # Get registration from server
    registration = _get_registration(name, email, organization, type,
                                     usage, nih_funded)
    from .nag import install
    if not install(session, registration):
        # Do not join mailing lists if we cannot install registration data
        return
    session.logger.info(ThankYou)

    # Register for mailing lists
    if join_discussion:
        _subscribe(session, "discussion", DiscussionURL, name, email)
    if join_announcements:
        _subscribe(session, "announcements", AnnouncementsURL, name, email)

def _get_registration(name, email, organization, org_type, usage, nih_funded):
    from urllib.parse import urlencode
    from urllib.request import urlopen
    from xml.dom import minidom
    from xml.parsers.expat import ExpatError
    # Required fields
    params = {
        "action":"Register from ChimeraX",
        "user":name,
        "email":email,
    }
    # Optional fields
    if organization:
        params["organization"] = organization
    if org_type:
        params["type"] = org_type
    if usage:
        params["usage"] = usage
    if nih_funded:
        params["nih"] = nih_funded
    with urlopen(RegistrationURL, urlencode(params).encode()) as f:
        text = f.read()
    try:
        dom = minidom.parseString(text)
    except ExpatError:
        raise UserError("Registration failed.  Please try again later.")
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
        "fullname":name,
        "email":email,
    }
    try:
        with urlopen(url, urlencode(params).encode()) as f:
            text = f.read()
        session.logger.info("%s is subscribed to the ChimeraX %s list" %
                            (email, label))
    except URLError as e:
        session.logger.warning("Failed to subscribed %s to the ChimeraX "
                               "%s list: %s" % (email, label, str(e)))


register_desc = CmdDesc(keyword=[("name", StringArg),
                                 ("email", StringArg),
                                 ("organization", StringArg),
                                 ("type", EnumOf(OrganizationTypes)),
                                 ("usage", EnumOf(UsageTypes)),
                                 ("nih_funded", BoolArg),
                                 ("join_discussion", BoolArg),
                                 ("join_announcements", BoolArg)],
                        required_arguments=[
                                  "name",
                                  "email",
                                  "organization",
                                  "type",
                                  "usage",
                                  "nih_funded"])
