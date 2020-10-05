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

def set_attr(session, objects, target, attr_name, attr_value, create=False):
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

    if attr_name.lower().endswith("color"):
        if not isinstance(attr_value, str):
            raise UserError("Trouble parsing shortened color name; please use full name")
        from chimerax.core.commands import ColorArg
        try:
            value = ColorArg.parse(attr_value, session)[0].uint8x4()
        except Exception as e:
            raise UserError(str(e))
    else:
        value = attr_value

    from chimerax.atomic.molarray import Collection
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
            if not session.attr_registration.has_attribute(items.object_class, attr_name):
                if create:
                    register_attr(session, items.object_class, attr_name, type(value))
                else:
                    raise UserError("Not creating attribute '%s'; use 'create true' to override" % attr_name)
            for item in items:
                setattr(item, attr_name, value)
    else:
        class_ = items[0].__class__
        if not session.attr_registration.has_attribute(class_, attr_name):
            if create:
                register_attr(session, class_, attr_name, type(value))
            else:
                raise UserError("Not creating attribute '%s'; use 'create true' to override" % attr_name)
        for item in items:
            attempt_set_attr(item, attr_name, value, attr_name, attr_value)

def attempt_set_attr(item, attr_name, value, orig_attr_name, value_string):
    try:
        setattr(item, attr_name, value)
    except Exception:
        try:
            setattr(item, attr_name, value_string)
        except Exception:
            from chimerax.core.errors import UserError
            raise UserError("Cannot set attribute '%s' to '%s'" % (orig_attr_name, value_string))

def register_attr(session, class_obj, attr_name, attr_type):
    if hasattr(class_obj, 'register_attr'):
        from chimerax.atomic.attr_registration import RegistrationConflict
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
    from chimerax.core.commands import register, CmdDesc, ObjectsArg
    from chimerax.core.commands import EmptyArg, Or, StringArg, BoolArg, IntArg, FloatArg
    desc = CmdDesc(required=[('objects', Or(ObjectsArg, EmptyArg)),
                            ('target', StringArg),
                            ('attr_name', StringArg),
                            ('attr_value', Or(IntArg, FloatArg, BoolArg, StringArg))],
                   keyword=[('create', BoolArg)],
                   synopsis="set attributes")
    register('setattr', desc, set_attr, logger=logger)
