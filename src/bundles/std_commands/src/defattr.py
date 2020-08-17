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

def cmd_defattr(session, structures, file_name, *, log=False):
    try:
        defattr(session, file_name, log=log, restriction=structures)
    except SyntaxError as e:
        from chimerax.core.errors import UserError
        raise UserError(str(e))

def defattr(session, file_name, *, log=False, restriction=None):
    """define attributes on objects

    Parameters
    ----------
    file_name : string
      Input file in 'defattr' format
    log : bool
      Whether to log assignment info
    restriction : Structures Collection or None
      If not None, structures to restrict the assignments to
      (in addition to any restrictions in the defattr file)
    """

    if restriction is None:
        from chimerax.atomic import all_structures
        restriction = all_structures(session)

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
    with open_input(file_name, encoding="utf-8") as f:
        data = []
        attrs = {}
        for lnum, raw_line in enumerate(f):
            # spaces in values could be significant, so instead of stripping just drop the '\n'
            # (which all line endings are translated to if newline=None [default] for open())
            line = raw_line[:-1]
            if not line.strip() or line[0] == '#':
                continue

            if line[0] == '\t':
                # data line
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
            if name in attrs:
                # presumably another set of control/data lines starting
                append_all_info(attrs, data, lnum+1)
                attrs = {}
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
        append_all_info(attrs, data, lnum+1)

    for attr_info, data_info in all_info:
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
        try:
            pre_existing_attr = getattr(recip_class, attr_name)
        except AttributeError:
            pass
        else:
            if callable(pre_existing_attr):
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
                    if pre_existing_attr:
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

def register_command(logger):
    from chimerax.core.commands import register, CmdDesc
    from chimerax.core.commands import EmptyArg, Or, OpenFileNameArg, BoolArg
    from chimerax.atomic import StructuresArg
    desc = CmdDesc(required=[('structures', Or(StructuresArg, EmptyArg)),
                            ('file_name', OpenFileNameArg),],
                   keyword=[('log', BoolArg)],
                   synopsis="define attributes in bulk")
    register('defattr', desc, cmd_defattr, logger=logger)
