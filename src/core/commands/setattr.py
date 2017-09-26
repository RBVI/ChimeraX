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
      Whether to create the attribute if the object doesn't already have it
    """
    if objects is None:
        from . import all_objects
        objects = all_objects(session)
    atoms = objects.atoms

    from ..errors import UserError
    if "atoms".startswith(target):
        items = atoms
        target = "atoms"
    elif "residues".startswith(target):
        items = atoms.unique_residues
        target = "residues"
    elif "chains".startswith(target):
        items = atoms.residues.unique_chains
    elif "structures".startswith(target):
        if "surfaces".startswith(target):
            raise UserError("Must provide enough attribute-target characters to distinguish"
                " 'structures' from 'surfaces'")
        items = atoms.unique_structures
        target = "structures"
    elif "models".startswith(target):
        items = objects.models
        target = "models"
    elif "bonds".startswith(target):
        items = atoms.intra_bonds
    elif "pseudobonds".startswith(target):
        if atoms:
            items = atoms.intra_pseudobonds
        else:
            pb_grps = []
            from ..atomic import PseudobondGroup, concatenate
            for m in objects.models:
                if isinstance(m, PseudobondGroup):
                    pb_grps.append(m)
            if pb_grps:
                items = concatenate([g.pseudobonds for g in pb_grps])
            else:
                items = None
        target = "pseudobonds"
    elif "groups".startswith(target):
        if atoms:
            items = atoms.intra_pseudobonds.unique_groups
        else:
            items = []
            from ..atomic import PseudobondGroup, concatenate
            for m in objects.models:
                if isinstance(m, PseudobondGroup):
                    items.append(m)
        target = "groups"
    elif "surfaces".startswith(target):
        from . import MolecularSurface
        items = [m for m in objects.models if isinstance(m, MolecularSurface)]
        target = "surfaces"
    else:
        raise UserError("Unknown attribute target: '%s'" % target)
    if not items:
        raise UserError("No items of type '%s' found" % target)

    if attr_name.lower().endswith("color"):
        if not isinstance(attr_value, str):
            raise UserError("Trouble parsing shortened color name; please use full name")
        from . import ColorArg
        try:
            value = ColorArg.parse(attr_value, session)[0].uint8x4()
        except Exception as e:
            raise UserError(str(e))
    else:
        value = attr_value

    from ..atomic.molarray import Collection
    if isinstance(items, Collection):
        from . import plural_of
        attr_names = plural_of(attr_name)
        if hasattr(items, attr_names):
            attempt_set_attr(items, attr_names, value, attr_name, attr_value)
        elif create:
            for item in items:
                setattr(item, attr_name, value)
        else:
            instances = items.instances(instantiate=False)
            # First check if they all have the attr
            for inst in instances:
                if inst is None or not hasattr(inst, attr_name):
                    raise UserError("Not creating attribute '%s'; use 'create true' to override"
                        % attr_name)
            for inst in instances:
                setattr(inst, attr_name, value)
    else:
        if not create:
            # First check if they all have the attr
            for item in items:
                if not hasattr(item, attr_name):
                    raise UserError("Not creating attribute '%s'; use 'create true' to override"
                        % attr_name)
        for item in items:
            attempt_set_attr(item, attr_name, value, attr_name, attr_value)

def attempt_set_attr(item, attr_name, value, orig_attr_name, value_string):
    try:
        setattr(item, attr_name, value)
    except:
        try:
            setattr(item, attr_name, value_string)
        except:
            from chimerax.core.errors import UserError
            raise UserError("Cannot set attribute '%s' to '%s'" % (orig_attr_name, value_string))

# -----------------------------------------------------------------------------
#
def register_command(session):
    from . import register, CmdDesc, ObjectsArg
    from . import EmptyArg, Or, StringArg, BoolArg, IntArg, FloatArg
    from ..map import MapArg
    desc = CmdDesc(required=[('objects', Or(ObjectsArg, EmptyArg)),
                            ('target', StringArg),
                            ('attr_name', StringArg),
                            ('attr_value', Or(BoolArg, IntArg, FloatArg, StringArg))],
                   keyword=[('create', BoolArg)],
                   synopsis="set attributes")
    register('setattr', desc, set_attr, logger=session.logger)
