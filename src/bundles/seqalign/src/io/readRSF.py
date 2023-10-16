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

"""
reads a GCG9 RSF file
"""

from chimerax.atomic import Sequence
from ..parse import FormatSyntaxError, make_readable

def read(session, f):
    IN_HEADER, START_ATTRS, IN_ATTRS, IN_FEATURES, IN_SEQ = range(5)

    state = IN_HEADER

    sequences = []
    line_num = 0
    has_offset = False
    longest = None
    file_attrs = {}
    for line in f.readlines():
        line = line.rstrip() # remove trailing whitespace/newline
        line_num += 1
        if line_num == 1:
            if line.startswith("!!RICH_SEQUENCE"):
                continue
            raise FormatSyntaxError("First line does not start with !!RICH_SEQUENCE")

        if state == IN_HEADER:
            if line.strip() == "..":
                state = START_ATTRS
                continue
            if "comments" in file_attrs:
                file_attrs["comments"] += "\n" + line
            else:
                file_attrs["comments"] = line
            continue
        if not line.strip():
            continue

        if state == START_ATTRS:
            if line.strip() == "{":
                state = IN_ATTRS
                cur_attr = None
                attrs = {}
            elif line:
                raise FormatSyntaxError(
                    "Unexpected text before start of sequence on line %d" &line_num)
            continue

        if state == IN_ATTRS or state == IN_FEATURES:
            if line.strip() == "sequence" and line[0] == "s":
                if "RSF name" not in attrs:
                    raise FormatSyntaxError("Sequence on line %d has no name" & line_num)
                state = IN_SEQ
                seq = Sequence(name=make_readable(attrs["RSF name"]))
                del attrs["RSF name"]
                seq.attrs = attrs
                if "RSF descrip" in attrs:
                    attrs["description"] = attrs["RSF descrip"]
                    del attrs["RSF descrip"]
                sequences.append(seq)
                if "RSF offset" in attrs:
                    seq.extend("." * int(attrs["RSF offset"]))
                    has_offset = True
                    del attrs["RSF offset"]
                continue
            if line.startswith("feature"):
                if state == IN_ATTRS:
                    attrs["RSF features"] = [[line[8:]]]
                else:
                    attrs["RSF features"].append([line[8:]])
                state = IN_FEATURES
                continue

        if state == IN_ATTRS:
            if line[0].isspace():
                # continuation
                if not cur_attr:
                    raise FormatSyntaxError("Bogus indentation at line %d" % line_num)
                if attrs[cur_attr]:
                    attrs[cur_attr] += "\n" + line
                else:
                    attrs[cur_attr] = line
                continue
            if " " in line.strip():
                cur_attr, val = line.split(None, 1)
                cur_attr.replace("_", " ")
                cur_attr = "RSF " + cur_attr
                attrs[cur_attr] = val.strip()
            else:
                cur_attr = "RSF " + line.strip().replace("_", " ")
                attrs[cur_attr] = ""
            continue

        if state == IN_FEATURES:
            attrs["RSF features"][-1].append(line)
            continue
        if line.strip() == "}":
            state = START_ATTRS
            if not longest:
                longest = len(seq)
            else:
                if len(seq) < longest:
                    seq.extend("." * (longest - len(seq)))
                elif len(seq) > longest:
                    longest = len(seq)
                    for s in sequences[:-1]:
                        s.extend("." * (longest - len(s)))
            continue
        seq.extend(line.strip())
        if not seq[0].isalpha():
            has_offset = True

    f.close()
    if state == IN_HEADER:
        raise FormatSyntaxError("No end to header (i.e. '..' line) found")
    if state == IN_ATTRS or state == IN_FEATURES:
        raise FormatSyntaxError("No sequence data found for sequence %s" % attrs["RSF name"])
    if state == IN_SEQ:
        raise FormatSyntaxError("No terminating brace for sequence %s" % attrs["RSF name"])
    if not has_offset:
        session.logger.warning("No offset fields in RSF file; assuming zero offset")
    return sequences, file_attrs, {}
