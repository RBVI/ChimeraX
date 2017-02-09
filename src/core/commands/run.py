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

def run(session, text, *, log=True, downgrade_errors=False):
    """execute a textual command

    Parameters
    ----------
    text : string
        The text of the command to execute.
    log : bool
        Print the command text to the reply log.
    downgrade_errors : bool
        True if errors in the command should be logged as informational.
    """

    from . import cli
    from ..errors import UserError
    command = cli.Command(session)
    try:
        return command.run(text, log=log)
    except UserError as err:
        if downgrade_errors:
            session.logger.info(str(err))
        else:
            session.logger.error(str(err))


def register_command(session):
    from . import CmdDesc, register, StringArg, BoolArg
    desc = CmdDesc(required=[('text', StringArg)],
                   optional=[('log', BoolArg),
                             ('downgrade_errors', BoolArg),
                         ],
                   synopsis='indirectly run a command')
    register('run', desc, run)

def concise_model_spec(session, models):
    model_ids = _form_id_dict(models)
    all_ids = _form_id_dict(session.models)
    _compact_fully_selected(model_ids, all_ids)
    _compact_identical_partials(model_ids)
    spec = _range_strings(model_ids, joiner=' #')
    # a lone '#' doesn't select everything for some reason, so...
    return '#' + spec if spec else ""

def _form_id_dict(models):
    ids = {}
    for model in models:
        if model.id is None:
            continue
        id_dict = ids
        for part in model.id:
            id_dict = id_dict.setdefault(part, {})
    return ids

def _compact_fully_selected(model_ids, all_ids):
    if model_ids == all_ids:
        model_ids.clear()
        return
    for part, part_dict in model_ids.items():
        _compact_fully_selected(part_dict, all_ids[part])

class IDRange:
    def __init__(self, _id):
        self.ids = [_id]

    def append(self, _id):
        self.ids.append(_id)

    def __str__(self):
        self.ids.sort()
        def append_range(ids_str, range_start, range_end):
            if ids_str:
                ids_str += ","
            if range_start == range_end:
                ids_str += str(range_start)
            else:
                ids_str += str(range_start) + '-' + str(range_end)
            return ids_str
        ids_str = ""
        prev = None
        for id_num in self.ids:
            if prev is None:
                range_start = range_end = id_num
            elif id_num == prev+1:
                range_end = id_num
            else:
                ids_str = append_range(ids_str, range_start, range_end)
                range_start = range_end = id_num
            prev = id_num
        return append_range(ids_str, range_start, range_end)

def _compact_identical_partials(model_ids):
    partials = {}
    for part, part_ids in model_ids.items():
        compacted = _compact_identical_partials(part_ids)
        for id_range, partial in partials.items():
            if compacted == partial:
                id_range.append(part)
                break
        else:
            partials[IDRange(part)] = compacted
    model_ids.clear()
    model_ids.update(partials)
    return model_ids

def _range_strings(id_ranges, joiner = ','):
    return joiner.join([_part_strings(id_range, partial) for id_range, partial in id_ranges.items()])

def _part_strings(id_range, subpart):
    if not subpart:
        return str(id_range)
    return str(id_range) + '.' + _range_strings(subpart)
