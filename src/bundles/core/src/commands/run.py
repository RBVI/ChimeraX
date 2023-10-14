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
import json
from chimerax.core.errors import UserError

def run(session, text, *, log=True, downgrade_errors=False, return_json=False, return_list=False):
    """execute a textual command

    Parameters
    ----------
    text : string
        The text of the command to execute.
    log : bool
        Print the command text to the reply log.
    downgrade_errors : bool
        True if errors in the command should be logged as informational.
    return_json : bool
        If True, underlying commands that themselves support a 'return_json' keyword should return a
        JSONResult object.
    return_list : bool
        True if a list should be returned even if only one command is executed
    """

    from chimerax.core.commands import Command
    from chimerax.core.errors import UserError
    command = Command(session)
    try:
        results = command.run(text, log=log, return_json=return_json)
    except UserError as err:
        if downgrade_errors:
            session.logger.info(str(err))
        else:
            raise
        results = []
    if return_list:
        return results
    return results[0] if len(results) == 1 else results


class JSONResult:
    """Class that should be returned by commands that support returning JSON (i.e. the command's
       function has a 'return_json' keyword, and has been called with that keyword set to True).
       Simply construct this class with the JSON string and the normal Python return value as the
       two constructor arguments (the latter could be None).
    """
    def __init__(self, json_value, python_value):
        self.json_value = json_value
        self.python_value = python_value


class ArrayJSONEncoder(json.JSONEncoder):
    """A version of json.JSONEncoder that can also encode numpy and tinyarray arrays"""

    def __init__(self, *args, **kw):
        self._user_default = kw.get('default', None)
        kw['default'] = self._encode_array
        super().__init__(*args, **kw)

    def _encode_array(self, val):
        # Does this look like an array?
        if hasattr(val, "__len__") and hasattr(val, "shape"):
            return [self._translate(v) for v in val]
        if self._user_default is None:
            raise TypeError("Can't JSON-encode '%s'" % repr(val))
        return self._user_default(val)

    def _translate(self, val):
        if hasattr(val, "__len__"):
            return [self._translate(v) for v in val]
        import numpy
        if isinstance(val, numpy.number):
            if isinstance(val, numpy.integer):
                return int(val)
            return float(val)
        return val

def concise_model_spec(session, models, relevant_types=None, allow_empty_spec=True):
    """For commands where the spec will be automatically narrowed down to specific types of models
       (e.g. command uses AtomicStructureArg rather than ModelsArgs), providing the 'relevant_types'
       arg (e.g. relevant_types=AtomicStructure) may allow a more concise spec to be generated.
       The 'models' arg will be pruned down to only those types.  If allow_empty_spec is True
       and all open models are to be specified then the empty string is returned.  If allow_empty_spec
       is False then a non-empty spec will always be returned.
    """
    universe = set(session.models if relevant_types is None else [x for x in session.models
                   if isinstance(x, relevant_types)])
    if relevant_types:
        models = [m for m in models if isinstance(m, relevant_types)]
    models = [m for m in models if m.id is not None]
    if not models:
        return '#'
    # algorithm only works (namely _make_id_tree) if ids are longest to shortest...
    universe = sorted(universe, key=lambda m: len(m.id))
    models = sorted(models, key=lambda m: len(m.id))
    u_id_tree = _make_id_tree(universe)
    m_id_tree = _make_id_tree(models)

    if allow_empty_spec and u_id_tree == m_id_tree:
        # for some reason '#' doesn't select all models, so...
        return ""

    full_idents = []
    for ident in m_id_tree.keys():
        u_subtree = u_id_tree[ident]
        m_subtree = m_id_tree[ident]
        full_idents.extend(_traverse_tree([ident], m_subtree, u_subtree))
    # full idents that end in '0' have submodels under them that should be excluded,
    # so handle them separately
    hash_idents = []
    hash_bang_idents = []
    for ident in full_idents:
        if ident[-1] == 0:
            hash_bang_idents.append(ident[:-1])
        else:
            hash_idents.append(ident)
    full_spec = ""
    for prefix, idents in [('#', hash_idents), ('#!', hash_bang_idents)]:
        if not idents:
            continue
        idents.sort(key=lambda x: ((len(x), x)))
        spec = prefix
        ident1 = ident2 = idents[0]
        show_full = True
        for ident in idents[1:]:
            if len(ident) != len(ident1) or ident[:-1] != ident2[:-1]:
                spec += _make_spec(ident1, ident2, show_full) + prefix
                ident1 = ident2 = ident
                show_full = True
            else:
                if ident[-1] == ident2[-1] + 1:
                    ident2 = ident
                else:
                    spec += _make_spec(ident1, ident2, show_full) + ','
                    ident1 = ident2 = ident
                    show_full = False
        spec += _make_spec(ident1, ident2, show_full)
        full_spec += spec
    return full_spec if full_spec else '#'


class NoneSelectedError(UserError):
    pass


def sel_or_all(session, sel_type_info, *, sel="sel", restriction=None, **concise_kw):
    # sel_type_info is either a list of strings each of which is appropriate as an arg for
    # session.selection.items(arg), or a Model subclass

    if type(sel_type_info) == type:
        # presumably Model subclass
        type_models = [m for m in session.models.list(type=sel_type_info)]
        if not type_models:
            raise NoneSelectedError("No %s models open" % sel_type_info.__name__)
        msel = [m for m in type_models if m.selected and m.visible]
        if msel:
            concise_spec = concise_model_spec(session, msel, **concise_kw)
            if restriction:
                if concise_spec:
                    return '(%s & %s)' % (concise_spec, restriction)
                return restriction
            return concise_spec
        shown_sel_models = [m for m in session.selection.models() if m.visible]
        if shown_sel_models:
            raise NoneSelectedError("No visible %s models selected" % sel_type_info.__name__)
        mshown = [m for m in type_models if m.visible]
        if not mshown:
            raise NoneSelectedError("No visible %s models" % sel_type_info.__name__)
        concise_spec = concise_model_spec(session, mshown, **concise_kw)
        if restriction:
            if concise_spec:
                return '(%s & %s)' % (concise_spec, restriction)
            return restriction
        return concise_spec

    # specific types rather than a Model subclass
    #
    # need to ask just _visible_ models for selected items, so can't use session.selection.selected_items()
    from . import commas
    from chimerax.core.models import Model
    sel_models = [m for m in session.selection.models() if m.__class__ != Model] # exclude grouping models
    shown_sel_models = [m for m in sel_models if m.visible]
    if shown_sel_models:
        relevant_sel_models = set()
        for sel_type in sel_type_info:
            for m in shown_sel_models:
                if m.selected_items(sel_type):
                    relevant_sel_models.add(m)
        if relevant_sel_models:
            if len(sel_models) == len(relevant_sel_models):
                if restriction:
                    return '(%s & %s)' % (sel, restriction)
                return sel
            else:
                concise_spec = concise_model_spec(session, relevant_sel_models, **concise_kw)
                if restriction:
                    return '(%s & %s & %s)' % (concise_spec, sel, restriction)
                return '(%s & %s)' % (concise_spec, sel)
        raise NoneSelectedError("No visible %s selected" % commas(sel_type_info))
    shown_models = [m for m in session.models if m.visible]
    if shown_models:
        concise_spec = concise_model_spec(session, shown_models, **concise_kw)
    else:
        raise NoneSelectedError("No visible models!")
    if restriction:
        if concise_spec:
            return '(%s & %s)' % (concise_spec, restriction)
        return restriction
    return concise_spec

def _make_id_tree(models):
    tree = {}
    for m in models:
        subtree = tree
        for i, ident in enumerate(m.id):
            subtree = subtree.setdefault(ident, {0: i == len(m.id) - 1})
    return tree

def _traverse_tree(prefix, m_tree, u_tree):
    if m_tree == u_tree:
        return [prefix]
    idents = []
    for ident in m_tree.keys():
        if ident == 0 and not m_tree[ident]:
            continue
        m_subtree = m_tree[ident]
        u_subtree = u_tree[ident]
        if isinstance(m_subtree, bool):
            if m_subtree:
                idents.append(prefix + [ident])
        elif m_subtree == u_subtree:
            idents.append(prefix + [ident])
        else:
            idents.extend(_traverse_tree(prefix + [ident], m_subtree, u_subtree))
    return idents

def _make_spec(ident1, ident2, show_full):
    if show_full:
        full1 = '.'.join([str(x) for x in ident1])
        if ident1 == ident2:
            return full1
        else:
            return "%s-%d" % (full1, ident2[-1])
    if ident1 == ident2:
        return "%d" % ident1[-1]
    return "%d-%d" % (ident1[-1], ident2[-1])
