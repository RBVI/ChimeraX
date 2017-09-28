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


def register(session, name, email, organization, type, usage, nih_funded,
             join_discussion=False, join_announcements=True):
    print("Register ChimeraX")
    print("Name:", name)
    print("E-Mail:", email)
    print("Organization:", organization)
    print("Organization type:", type)
    print("Primary usage:", usage)
    print("NIH-funded:", nih_funded)
    print("Join discussion mailing list:", join_discussion)
    print("Join announcements mailing list:", join_announcements)

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
