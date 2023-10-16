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

from chimerax.core.errors import UserError

match_modes = ["any", "non-zero", "1-to-1"]

def cmd_defattr(session, structures, file_name, *, log=False):
    session.logger.warning("The 'defattr' command is deprecated."
        "  Just use the 'open' command with your .defattr file.")
    try:
        defattr(session, file_name, log=log, restriction=structures)
    except SyntaxError as e:
        raise UserError(str(e))

def defattr(session, data, *, log=False, restriction=None, file_name=None, summary=True):
    """define attributes on objects

    Parameters
    ----------
    data : file name or stream
      Input file in 'defattr' format
    log : bool
      Whether to log assignment info
    restriction : list/Collection of Structures or None
      If not None, structures to restrict the assignments to
      (in addition to any restrictions in the defattr file)
    file_name : string
      If data is a stream, the original file name can be provided here
    """

    if restriction is None:
        from chimerax.atomic import all_structures
        restriction = all_structures(session)
    else:
        from chimerax.atomic import Collection, Structures
        if not isinstance(restriction, Collection):
            restriction = Structures(restriction)

    control_defaults = {
        'match mode': "any",
        'recipient': "atoms",
        'none handling': "None"
    }
    from chimerax.atomic import Atom, Bond, Pseudobond, Residue, Chain, Structure
    recipient_info = {
        "atoms": (Atom, lambda objs: objs.atoms),
        "bonds": (Bond, lambda objs: objs.bonds),
        "pseudobonds": (Pseudobond, lambda objs: objs.pseudobonds),
        "residues": (Residue, lambda objs: objs.residues),
        "chains": (Chain, lambda objs: objs.chains),
        # since we always restrict to structures, can just use Objects.models()
        "molecules": (Structure, lambda objs: objs.models),
        "structures": (Structure, lambda objs: objs.models),
    }
    legal_control_values = {
        'match mode': set(["any", "non-zero", "1-to-1"]),
        'recipient': set(recipient_info.keys()),
        'none handling': set(["None", "string", "delete"])
    }
    if file_name is None:
        file_name = data if isinstance(data, str) else getattr(data, 'name', "unknown file")
    all_info = []
    def append_all_info(attr_info, data_info, line_num, *, ai=all_info, fn=file_name):
        if 'attribute' not in attr_info:
            raise SyntaxError("No attribute name defined for data lines %d and earlier in %s"
                % (line_num, fn))
        if not data_info:
            raise SyntaxError("No data lines for attribute '%s' in %s" % (attr_info['attribute'], fn))
        ai.append((attr_info, data_info))
    from chimerax.core.commands import AtomSpecArg, AttrNameArg, AnnotationError, NoneArg, ColorArg, commas
    from chimerax.core.commands import IntArg, FloatArg
    from chimerax.io import open_input
    with open_input(data, encoding="utf-8") as f:
        data = []
        attrs = {}
        empty_file = True
        for lnum, raw_line in enumerate(f):
            empty_file = False
            # spaces in values could be significant, so instead of stripping just drop the '\n'
            # (which all line endings are translated to if newline=None [default] for open())
            line = raw_line[:-1]
            if not line.strip() or line[0] == '#':
                continue

            if line[0] == '\t':
                # data line
                if 'attribute' not in attrs:
                    raise SyntaxError("'attribute' control line must precede any data lines")
                datum = line[1:].split('\t')
                if len(datum) != 2:
                    raise SyntaxError("Data line %d in %s not of the form: <tab> atomspec <tab> value"
                        % (lnum+1, file_name))
                data.append((lnum+1, *datum))
                continue
            # control line
            try:
                name, value = line.split(": ")
            except ValueError:
                raise SyntaxError("Line %d in %s is either not of the form 'name: value'"
                    " or is missing initial tab" % (lnum+1, file_name))
            name = name.strip().lower()
            value = value.strip()
            if data:
                # control change; stow current controls/data
                append_all_info(attrs, data, lnum+1)
                data = []
            if name == 'attribute':
                try:
                    final_value, *args = AttrNameArg.parse(value, session)
                except AnnotationError as e:
                    raise SyntaxError("Bad attribute name ('%s') given on line %d of %s: %s"
                        % (value, lnum+1, file_name, str(e)))
            elif name not in legal_control_values:
                raise SyntaxError("Unrecognized control type ('%s') given on line %d of %s"
                    % (name, lnum+1, file_name))
            elif value not in legal_control_values[name]:
                raise SyntaxError("Illegal control value ('%s') for %s given on line %d of %s; legal"
                    " values are: %s" % (value, name, lnum+1, file_name, commas(legal_control_values[name])))
            else:
                final_value = value
            attrs[name] = final_value
        if empty_file:
            raise SyntaxError("%s is empty" % file_name)
        else:
            append_all_info(attrs, data, lnum+1)

    for attr_info, data_info in all_info:
        num_assignments = 0
        attr_name = attr_info['attribute']
        color_attr = attr_name.lower().endswith('color') or attr_name.lower().endswith('colour')

        match_mode = attr_info.get('match mode', control_defaults['match mode'])

        none_handling = attr_info.get('none handling', control_defaults['none handling'])
        none_okay = none_handling != 'string'
        none_seen = False
        eval_vals = ["true", "false"]
        if none_okay:
            eval_vals.append("none")

        recipient = attr_info.get('recipient', control_defaults['recipient'])
        recip_class, instance_fetch = recipient_info[recipient]
        seen_types = set()
        is_builtin_attr = True
        try:
            builtin_attr = getattr(recip_class, attr_name)
        except AttributeError:
            is_builtin_attr = False
        else:
            if callable(builtin_attr):
                raise ValueError("%s is a method of the %s class and cannot be redefined"
                    % (attr_name, recip_class.__name__))
            if attr_name[0].isupper():
                raise ValueError("%s is a constant in the %s class and cannot be redefined"
                    % (attr_name, recip_class.__name__))

        for line_num, spec, value_string in data_info:
            try:
                atom_spec, *args = AtomSpecArg.parse(spec, session)
            except AnnotationError as e:
                raise SyntaxError("Bad atom specifier (%s) on line %d of %s" % (spec, line_num, file_name))

            try:
                objects = atom_spec.evaluate(session, models=restriction)
            except Exception as e:
                raise SyntaxError("Error evaluating atom specifier (%s) on line %d of %s: %s"
                    % (spec, line_num, file_name, str(e)))

            matches = instance_fetch(objects)

            if not matches and match_mode != "any":
                raise SyntaxError("Selector (%s) on line %d of %s matched nothing"
                    % (spec, line_num, file_name))
            if len(matches) > 1 and match_mode == "1-to-1":
                raise SyntaxError("Selector (%s) on line %d of %s matched multiple %s"
                    % (spec, line_num, file_name, recipient))
            num_assignments += len(matches)

            if log:
                session.logger.info("Selector %s matched %s"
                    % (spec, commas([str(x) for x in matches], conjunction="and")))

            if not value_string:
                raise SyntaxError("No data value on line %d of %s" % (line_num, file_name))

            # Can't just use normal argument parsers willy nilly since strings are allowed to have
            # leading/trailing whitespace, don't want to accept shorten ed forms of booleans, etc.
            if color_attr:
                try:
                    value, text, rest = ColorArg.parse(value_string, session)
                    if rest:
                        raise AnnotationError("trailing text")
                    seen_types.add("color")
                    value = value.uint8x4()
                except AnnotationError:
                    if none_okay:
                        try:
                            value, text, rest = NoneArg.parse(value_string, session)
                            if rest:
                                raise AnnotationError("trailing text")
                            seen_types.add(None)
                        except AnnotationError:
                            raise SyntaxError("Value (%s) on line %d of %s is not recognizable as either a"
                                " color value or None" % (value_string, line_num, file_name))
                    else:
                        raise SyntaxError("Value (%s) on line %d of %s is not recognizable as a color value"
                            % (value_string, line_num, file_name))
            else:
                if value_string.strip() != value_string:
                    value = value_string
                    seen_types.add(str)
                elif value_string.startswith('"') and value_string.endswith('"'):
                    value = value_string[1:-1]
                    seen_types.add(str)
                elif value_string.lower() in eval_vals:
                    value = eval(value_string.capitalize())
                    if value is None:
                        seen_types.add(None)
                    else:
                        seen_types.add(bool)
                else:
                    try:
                        value, text, rest = IntArg.parse(value_string, session)
                        if rest:
                            raise AnnotationError("trailing text")
                        seen_types.add(int)
                    except AnnotationError:
                        try:
                            value, text, rest = FloatArg.parse(value_string, session)
                            if rest:
                                raise AnnotationError("trailing text")
                            seen_types.add(float)
                        except AnnotationError:
                            value = value_string
                            seen_types.add(str)

            for match in matches:
                if value is not None or none_handling == "None":
                    setattr(match, attr_name, value)
                elif hasattr(match, attr_name):
                    if is_builtin_attr:
                        raise RuntimeError("Cannot remove builtin attribute %s from class %s"
                            % (attr_name, recip_class.__name__))
                    else:
                        delattr(match, attr_name)

        can_return_none = None in seen_types
        seen_types.discard(None)
        if len(seen_types) == 1:
            seen_type = seen_types.pop()
            attr_type = None if seen_type == "color" else seen_type
        elif seen_types == set([int, float]):
            attr_type = float
        else:
            attr_type = None
        recip_class.register_attr(session, attr_name, "defattr command", attr_type=attr_type,
            can_return_none=can_return_none)

        if summary:
            session.logger.info("Assigned attribute '%s' to %d %s using match mode: %s" % (attr_name,
                num_assignments, (recipient if num_assignments != 1 else recipient[:-1]), match_mode))

def parse_attribute_name(session, attr_name, *, allowable_types=None):
    from chimerax.atomic import Atom, Residue, Structure
    from chimerax.core.attributes import MANAGER_NAME, type_attrs
    if len(attr_name) > 1 and attr_name[1] == ':':
        attr_level = attr_name[0]
        if attr_level not in "arm":
            raise UserError("Unknown attribute level: '%s'" % attr_level)
        attr_name = attr_name[2:]
        class_obj = {'a': Atom, 'r': Residue, 'm': Structure}[attr_level]
        if allowable_types:
            allowable_attrs = session.get_state_manager(MANAGER_NAME).attributes_returning(
                class_obj, allowable_types, none_okay=True)
        else:
            allowable_attrs = type_attrs(class_obj)
        if attr_name not in allowable_attrs:
            raise UserError("Unknown/unregistered %s attribute %s" % (class_obj.__name__, attr_name))
    else:
        # try to find the attribute, in the order Atom->Residue->Structure
        for class_obj, attr_level in [(Atom, 'a'), (Residue, 'r'), (Structure, 'm')]:
            if allowable_types:
                allowable_attrs = session.get_state_manager(MANAGER_NAME).attributes_returning(
                    class_obj, allowable_types, none_okay=True)
            else:
                allowable_attrs = type_attrs(class_obj)
            if attr_name in allowable_attrs:
                break
        else:
            adjective = ""
            if allowable_types:
                if allowable_types == [bool]:
                    adjective = "boolean "
                elif allowable_types == [str]:
                    adjective = "string "
                elif allowable_types == [int]:
                    adjective = "integer "
                elif allowable_types == [float]:
                    adjective = "floating-point "
                elif set(allowable_types) == set([int, float]):
                    adjective = "numeric "
            raise UserError("No known/registered %sattribute %s" % (adjective, attr_name))
    return attr_name, class_obj

def write_defattr(session, output, *, models=None, attr_name=None, match_mode="1-to-1", model_ids=None,
            selected_only=False):
    """'attr_name' is the same as for "color byattr": it can be the plain attribute name or prefixed with
       'a:', 'r:' or 'm:' to indicate what "level" (atom, residue, model/structure) to look for the
       attribute.  If no prefix, then look in the order a->r->m until one is found.

       'model_ids' indicates whether the atom specifiers written should include the model component.
       'None' indicates that they should be included only if multiple structures are open.

       'match_mode' will be written into the defattr header section.

       If 'selected_only' is True, then only items that are also selected will be written.
    """
    if attr_name is None:
        raise UserError("Must specify an attribute name to save")

    from chimerax.atomic import Atom, Residue, Structure, concatenate
    if models is None:
        structures = session.models.list(type=Structure)
    else:
        structures = [m for m in models if isinstance(m, Structure)]

    # gather items whose attributes will be saved
    attr_name, class_obj = parse_attribute_name(session, attr_name)
    recipient = {Atom: 'atoms', Residue: 'residues', Structure: 'structures'}[class_obj]
    sources = []
    if selected_only:
        for s in structures:
            if recipient == "structures":
                if s.selected:
                    sources.append(s)
            else:
                sources.extend(s.selected_items(recipient))
    else:
        if recipient == "structures":
            sources = structures
        else:
            for s in structures:
                sources.append(getattr(s, recipient))
    if recipient != "structures":
        if sources:
            sources = concatenate(sources)

    none_handling = None
    type_warning_issued = False
    from chimerax import io
    num_saved = 0
    with io.open_output(output, 'utf-8') as stream:
        print("attribute: %s" % attr_name, file=stream)
        print("recipient: %s" % recipient, file=stream)
        print("match mode: %s" % match_mode, file=stream)
        for source in sources:
            try:
                val = getattr(source, attr_name)
            except AttributeError:
                continue
            if val is None:
                if none_handling != "python":
                    print("none handling: None", file=stream)
                    none_handling = "python"
                val = "None"
            elif type(val) == str:
                try:
                    float(val)
                except ValueError:
                    if val.lower() in ("true", "false"):
                        # Could be mis-interpreted as bool
                        val = '"%s"' % val
                    # string can't be misinterpreted as numeric, check for none/None
                    elif (val == "none" or val == "None") and none_handling != "string":
                        print("none handling: string", file=stream)
                        none_handling = "string"
                else:
                    val = '"%s"' % val
            elif type(val) != bool:
                # Can't just "not isinstance(val, (int, float))" because of numpy
                try:
                    float(str(val))
                except ValueError:
                    if not type_warning_issued:
                        session.logger.warning("One or more attribute values aren't integer, floating-point,"
                            " boolean, string or None (e.g. %s); skipping those" % repr(val))
                        type_warning_issued = True
                    continue
            if recipient == "structures":
                spec = source.atomspec
            else:
                spec = source.string(style="command",
                    omit_structure=(None if model_ids is None else not model_ids))
            print("\t%s\t%s" % (spec, str(val)), file=stream)
            num_saved += 1

        session.logger.info("Saved attribute '%s' of %d %s using match mode: %s to %s" % (attr_name,
            num_saved, (recipient if num_saved != 1 else recipient[:-1]), match_mode, output))

def register_command(logger):
    from chimerax.core.commands import register, CmdDesc
    from chimerax.core.commands import EmptyArg, Or, OpenFileNameArg, BoolArg
    from chimerax.atomic import StructuresArg
    desc = CmdDesc(required=[('structures', Or(StructuresArg, EmptyArg)),
                            ('file_name', OpenFileNameArg),],
                   keyword=[('log', BoolArg)],
                   synopsis="define attributes in bulk")
    register('defattr', desc, cmd_defattr, logger=logger)
