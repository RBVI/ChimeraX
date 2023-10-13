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

from chimerax.core.commands import Color8Arg, IntArg, FloatArg, BoolArg, StringArg

def set_attr(session, objects, target, attr_name, attr_value, create=False, type_=None):
    """set attributes on objects

    Parameters
    ----------
    objects : Objects
      Which objects to sat attributes on.
    target : string
      Indicates what kind of object to set attributes on.  Can be shortened to unique string.
      Possible values: atoms, residues, chains, structures, models, bonds, pseudobonds, groups,
      surfaces.
    attr_name : string
      Attribute name
    attr_value : string
      If attr_name.lower() ends in "color", will be treated as a color name, otherwise as
      whatever type it "looks like": int, bool, float, string.
    create : bool
      Whether to create the attribute if the class doesn't already have it
    type_ : Annotation (e.g. BoolArg) from chimerax.core.commands
      If None, heuristics will be used to guess the type of the attr_value.
    """
    if objects is None:
        from chimerax.core.objects import all_objects
        objects = all_objects(session)

    from chimerax.core.errors import UserError
    if "atoms".startswith(target):
        items = objects.atoms
        target = "atoms"
    elif "residues".startswith(target):
        items = objects.residues
        target = "residues"
    elif "chains".startswith(target):
        items = objects.residues.unique_chains
    elif "structures".startswith(target):
        if "surfaces".startswith(target):
            raise UserError("Must provide enough attribute-target characters to distinguish"
                " 'structures' from 'surfaces'")
        items = objects.residues.unique_structures
        target = "structures"
    elif "models".startswith(target):
        items = objects.models
        target = "models"
    elif "bonds".startswith(target):
        items = objects.bonds
    elif "pseudobonds".startswith(target):
        items = objects.pseudobonds
        target = "pseudobonds"
    elif "groups".startswith(target):
        items = objects.pseudobonds.unique_groups
        target = "groups"
    elif "surfaces".startswith(target):
        from chimerax.atomic import MolecularSurface
        items = [m for m in objects.models if isinstance(m, MolecularSurface)]
        target = "surfaces"
    else:
        raise UserError("Unknown attribute target: '%s'" % target)
    from chimerax.core.commands import plural_form
    match_msg = "Assigning %s attribute to %d %s" % (attr_name, len(items), plural_form(items, "item"))
    if not items:
        session.logger.info('<font color="OrangeRed">' +  match_msg + "</font>", is_html=True)
        return
    session.logger.info(match_msg)

    # Use None as the "string parser", since we don't want to lose a level of embedded quotes by parsing
    # the value a second time
    if type_ is None:
        if attr_name.lower().endswith("color"):
            parsers = [Color8Arg]
        elif attr_value in ('true', 'false'):
            parsers = [BoolArg]
        elif attr_value and attr_value[0] in ['"', "'"]:
            # prevent the value '3' (originally typed as "'3'") from evaluating to integer
            parsers = [None]
        else:
            parsers = [IntArg, FloatArg, None]
    else:
        if type_ == StringArg:
            parsers = [None]
        else:
            parsers = [type_]
    for parser in parsers:
        if parser is None:
            value = attr_value
            break
        try:
            val, cmd_text, remainder = parser.parse(attr_value, session)
        except Exception as e:
            if len(parsers) == 1:
                raise UserError(str(e))
        else:
            if not remainder:
                value = val
                break
    else:
        raise UserError("Could not determine type of value '%s'" % attr_value)

    from chimerax.atomic.molarray import Collection
    from chimerax.core.attributes import MANAGER_NAME
    if isinstance(items, Collection):
        from chimerax.core.commands import plural_of
        attr_names = plural_of(attr_name)
        if hasattr(items, attr_names):
            attempt_set_attr(items, attr_names, value, attr_name, attr_value)
        else:
            # if already registered, set values; if not, register if create==True
            # Since 'create' is more to prevent accidental attribute creation due to typos than
            # to strictly enforce typing (especially since int and float "conflict") check
            # existence first and only create if non-existent
            if not session.get_state_manager(MANAGER_NAME).has_attribute(items.object_class, attr_name):
                if create:
                    register_attr(session, items.object_class, attr_name, type(value))
                else:
                    raise UserError("Not creating attribute '%s'; use 'create true' to override" % attr_name)
            for item in items:
                setattr(item, attr_name, value)
        if items.object_class in session.change_tracker.tracked_classes:
            session.change_tracker.add_modified(items, attr_name + " changed")
    else:
        class_ = list(items)[0].__class__
        from chimerax.core.attributes import type_attrs
        if attr_name not in type_attrs(class_):
            if create:
                register_attr(session, class_, attr_name, type(value))
            else:
                raise UserError("Not creating attribute '%s'; use 'create true' to override" % attr_name)
        for item in items:
            attempt_set_attr(item, attr_name, value, attr_name, attr_value)
        # only need to add to change tracker if attribute isn't builtin
        if isinstance(item, tuple(session.change_tracker.tracked_classes)) \
        and session.get_state_manager(MANAGER_NAME).has_attribute(class_, attr_name):
            session.change_tracker.add_modified(items, attr_name + " changed")

def attempt_set_attr(item, attr_name, value, orig_attr_name, value_string):
    try:
        setattr(item, attr_name, value)
    except Exception:
        from chimerax.core.errors import UserError
        raise UserError("Cannot set attribute '%s' to '%s'" % (orig_attr_name, value_string))

def register_attr(session, class_obj, attr_name, attr_type):
    if hasattr(class_obj, 'register_attr'):
        from chimerax.core.attributes import RegistrationConflict
        try:
            class_obj.register_attr(session, attr_name, "setattr command", attr_type=attr_type)
        except RegistrationConflict as e:
            from chimerax.core.errors import LimitationError
            raise LimitationError(str(e))
    else:
        session.logger.warning("Class %s does not support attribute registration; '%s' attribute"
            " will not be preserved in sessions." % (class_obj.__name__, attr_name))

# -----------------------------------------------------------------------------
#
def register_command(logger):
    from chimerax.core.commands import register, CmdDesc, ObjectsArg, EmptyArg, Or, EnumOf
    desc = CmdDesc(required=[('objects', Or(ObjectsArg, EmptyArg)),
                            ('target', StringArg),
                            ('attr_name', StringArg),
                            ('attr_value', StringArg)],
                   keyword=[('create', BoolArg),
                            ('type_', EnumOf((Color8Arg, BoolArg, IntArg, FloatArg, StringArg),
                                            ("color", "boolean", "integer", "float", "string")))],
                   synopsis="set attributes")
    register('setattr', desc, set_attr, logger=logger)
