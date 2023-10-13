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
reads a Stockholm file
"""

from chimerax.atomic import Sequence
from ..parse import FormatSyntaxError, make_readable

# what format-specific features can be used generically; also used by saving code...
generic_file_attrs = {
    "AC": "accession",
    "AU": "author",
    "CC": "comments",
    "DE": "description"
}
generic_seq_attrs = {
    "AC": "accession",
    "DE": "description",
    "WT": "weight"
}

def read(session, f):
    line_num = 0
    file_attrs = {}
    file_markups = {}
    seq_attrs = {}
    seq_markups = {}
    sequences = {}
    seq_sequence = []
    for line in f.readlines():
        line = line.rstrip() # drop trailing newline/whitespace
        line_num += 1
        if line_num == 1:
            if line.startswith("# STOCKHOLM"):
                continue
            raise FormatSyntaxError("File does not start with '# STOCKHOLM'")
        if not line:
            continue
        if line.startswith('#='):
            markup_type = line[2:4]
            markup = line[5:].strip()
            def try_split(num_split):
                fields = markup.split(None, num_split)
                if len(fields) == num_split:
                    # value is empty
                    fields.append("")
                if len(fields) != num_split + 1:
                    raise FormatSyntaxError("Not enough arguments after #=%s markup on line %d"
                        % (markup_type, line_num))
                return fields
            if markup_type == "GF":
                tag, val = try_split(1)
                tag = tag.replace("_", " ")
                tag = generic_file_attrs.get(tag, "Stockholm " + tag)
                if tag in file_attrs:
                    file_attrs[tag] += '\n' + val
                else:
                    file_attrs[tag] = val
            elif markup_type == "GS":
                seq_name, tag, val = try_split(2)
                tag = tag.replace("_", " ")
                attrs = seq_attrs.setdefault(seq_name, {})
                tag = generic_seq_attrs.get(tag, "Stockholm " + tag)
                if tag in attrs:
                    attrs[tag] += '\n' + val
                else:
                    attrs[tag] = val
            elif markup_type == "GC":
                tag, val = try_split(1)
                tag = tag.replace("_", " ")
                file_markups[tag] = file_markups.get(tag, "") + val
            elif markup_type == "GR":
                seq_name, tag, val = try_split(2)
                tag = tag.replace("_", " ")
                seq_markups.setdefault(seq_name, {}).setdefault(tag, "")
                seq_markups[seq_name][tag] += val
            # ignore other types
            continue
        elif line.startswith('#'):
            # unstructured comment
            if 'comments' in file_attrs:
                file_attrs['comments'] += "\n" + line[1:]
            else:
                file_attrs['comments'] = line[1:]
            continue
        elif line.strip() == "//":
            # end of sequence alignment blocks, but comments may follow this, so keep going...
            continue
        # sequence info...
        try:
            seq_name, block = line.split(None, 1)
        except ValueError:
            raise FormatSyntaxError(
                "Sequence info not in name/contents format on line %d" % line_num)
        if seq_name not in sequences:
            sequences[seq_name] = Sequence(name=make_readable(seq_name))
            seq_sequence.append(seq_name)
        sequences[seq_name].extend(block)
    f.close()
    for seq_name, seq in sequences.items():
        if seq_name in seq_attrs:
            seq.attrs = seq_attrs[seq_name]
        if seq_name in seq_markups:
            seq.markups = seq_markups[seq_name]
            for tag, markup in seq.markups.items():
                if len(markup) != len(seq):
                    session.logger.warning("Markup %s for sequence %s is wrong length; ignoring"
                        % (tag, seq_name))
                    del seq.markups[tag]
    for seq_info, label in [(seq_attrs, "sequence"), (seq_markups, "residue")]:
        for seq_name in seq_info.keys():
            if seq_name in sequences:
                continue
            # might be sequence name if trailing '/start-end' is removed...
            for full_name in sequences.keys():
                if full_name.startswith(seq_name) \
                and full_name[len(seq_name)] == '/' \
                and '/' not in full_name[len(seq_name)+1:]:
                    break
            else:
                raise FormatSyntaxError("%s annotations provided for non-existent sequence %s"
                    % (label.capitalize(), seq_name))
            session.logger.info("Updating %s %s annotations with %s annotations"
                % (full_name, label, seq_name))
            seq_info[full_name].update(seq_info[seq_name])
            del seq_info[seq_name]
    for tag, markup in file_markups.items():
        if len(markup) != len(sequences[seq_sequence[0]]):
            raise FormatSyntaxError("Column annotation %s is wrong length" % tag)

    return [sequences[name] for name in seq_sequence], file_attrs, file_markups
