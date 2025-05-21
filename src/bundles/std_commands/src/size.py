# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2022 Regents of the University of California. All rights reserved.
# The ChimeraX application is provided pursuant to the ChimeraX license
# agreement, which covers academic and commercial uses. For more details, see
# <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
#
# This particular file is part of the ChimeraX library. You can also
# redistribute and/or modify it under the terms of the GNU Lesser General
# Public License version 2.1 as published by the Free Software Foundation.
# For more details, see
# <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
#
# THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
# EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
# LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
# VERSION 2.1
#
# This notice must be embedded in or attached to all copies, including partial
# copies, of the software or any revisions or derivations thereof.
# === UCSF ChimeraX Copyright ===

from chimerax.core.errors import UserError

def size(session, objects=None, atom_radius=None,
          stick_radius=None, pseudobond_radius=None, ball_scale=None, verbose=True):
    '''Set the sizes of atom and bonds.

    Parameters
    ----------
    objects : Objects
        Change the size of these atoms, bonds and pseudobonds.
        If not specified then all are changed.
    atom_radius : float, (bool, float) or "default"
      New radius value for atoms.  The optional boolean is whether the float is a delta.
    stick_radius : float or (bool, float)
      New radius value for bonds shown in stick style. The optional boolean is whether the float is a delta.
    pseudobond_radius : float or (bool, float)
      New radius value for pseudobonds. The optional boolean is whether the float is a delta.
    ball_scale : float or (bool, float)
      Multiplier times atom radius for determining atom size in ball style (default 0.3).
      The optional boolean is whether the float is a delta.
    '''
    if objects is None:
        from chimerax.core.objects import all_objects
        objects = all_objects(session)

    from chimerax.core.undo import UndoState
    undo_state = UndoState("size")
    what = []

    if atom_radius is not None:
        atoms = objects.atoms
        if atom_radius == 'default':
            undo_state.add(atoms, "radii", atoms.radii, atoms.default_radii)
            atoms.radii = atoms.default_radii
        else:
            if isinstance(atom_radius, tuple):
                is_delta, amount = atom_radius
            else:
                is_delta, amount = False, atom_radius
            if is_delta:
                old_radii = atoms.radii
                if amount < 0 and len(old_radii) > 0 and min(old_radii) <= abs(amount):
                    raise UserError("Cannot reduce atom radius to <= 0")
                atoms.radii += amount
                undo_state.add(atoms, "radii", old_radii, atoms.radii)
            else:
                if amount <= 0:
                    raise UserError('Atom radius must be greater than 0.')
                undo_state.add(atoms, "radii", atoms.radii, amount)
                atoms.radii = amount
        what.append('%d atom radii' % len(atoms))

    if stick_radius is not None:
        b = objects.bonds
        if isinstance(stick_radius, tuple):
            is_delta, amount = stick_radius
        else:
            is_delta, amount = False, stick_radius
        if is_delta:
            old_radii = b.radii
            if amount < 0 and len(old_radii) > 0 and min(old_radii) <= abs(amount):
                raise UserError("Cannot reduce stick radius to <= 0")
            b.radii += amount
            undo_state.add(b, "radii", old_radii, b.radii)
            # If singleton atom specified then set the single-atom stick radius.
            for s, atoms in objects.atoms.by_structure:
                if (atoms.num_bonds == 0).any():
                    if amount < 0 and s.bond_radius < amount:
                        raise UserError("Cannot reduce bond radius to <= 0")
                    s.bond_radius += amount
        else:
            if amount <= 0:
                raise UserError('Bond radius must be greater than 0.')
            undo_state.add(b, "radii", b.radii, amount)
            b.radii = amount
            # If singleton atom specified then set the single-atom stick radius.
            for s, atoms in objects.atoms.by_structure:
                if (atoms.num_bonds == 0).any():
                    s.bond_radius = amount
        what.append('%d bond radii' % len(b))

    if pseudobond_radius is not None:
        if isinstance(pseudobond_radius, tuple):
            is_delta, amount = pseudobond_radius
        else:
            is_delta, amount = False, pseudobond_radius
        pb = objects.pseudobonds
        if is_delta:
            old_radii = pb.radii
            if amount < 0 and len(old_radii) > 0 and min(old_radii) <= abs(amount):
                raise UserError("Cannot reduce pseudobond radius to <= 0")
            pb.radii += amount
            undo_state.add(pb, "radii", old_radii, pb.radii)
        else:
            if amount <= 0:
                raise UserError('Pseudobond radius must be greater than 0.')
            undo_state.add(pb, "radii", pb.radii, amount)
            pb.radii = amount
            from chimerax.atomic import concatenate
        what.append('%d pseudobond radii' % len(pb))

    if ball_scale is not None:
        if isinstance(ball_scale, tuple):
            is_delta, amount = ball_scale
        else:
            is_delta, amount = False, ball_scale
        mols = objects.residues.unique_structures
        if is_delta:
            for s in mols:
                if amount < 0 and s.ball_scale + amount <= 0:
                    raise UserError("Cannot reduce ball scale to <= 0")
                undo_state.add(s, "ball_scale", s.ball_scale, s.ball_scale + amount)
                s.ball_scale += amount
        else:
            if amount <= 0:
                raise UserError('Ball scale must be greater than 0.')
            for s in mols:
                undo_state.add(s, "ball_scale", s.ball_scale, amount)
                s.ball_scale = amount
        what.append('%d ball scales' % len(mols))

    if verbose and what:
        msg = 'Changed %s' % ', '.join(what)
        log = session.logger
        log.status(msg)
        log.info(msg)

    session.undo.register(undo_state)

from chimerax.core.commands import register, CmdDesc, ObjectsArg, EmptyArg, EnumOf, Or, FloatOrDeltaArg
from chimerax.core.commands import PositiveFloatArg, Annotation, FloatArg, StringArg, RepeatOf

AtomRadiiStyleArg = EnumOf(['sphere', 'ball', 'unchanged'])
AtomRadiiStyleArg.default = 'sphere'

def size_by_attr(session, attr_name, atoms=None, way_points=None, *, average=None, no_value_radius=None,
          undo_name="size byattribute", style=AtomRadiiStyleArg.default):
    '''
    Size atoms by attribute value using (attr-val, radius) way points.  Attr-val can be 'max' or 'min'
      to represent the maximum or minimum of that attribute value for the atoms.

    attr_name : string (actual Python attribute name optionally prefixed by 'a:'/'r:'/'m:'
      for atom/residue/model attribute. If no prefix, then the Atom/Residue/Structure classes
      will be searched for the attribute (in that order).
    atoms : Atoms
    '''
    from .defattr import parse_attribute_name
    attr_name, class_obj = parse_attribute_name(session, attr_name, allowable_types=[int, float])

    if atoms is None:
        from chimerax.atomic import all_atoms
        atoms = all_atoms(session)

    if len(atoms) == 0:
        session.logger.warning('No atoms specified')
        return

    if way_points is None:
        way_points = [('min', 1.0), ('max', 4.0)]
    elif len(way_points) < 2:
        raise UserError("Must specify at least 2 attr-val:radius pairs (or no pairs)")

    from chimerax.core.undo import UndoState
    undo_state = UndoState(undo_name)

    from chimerax.atomic import Atom, Residue
    if class_obj == Atom:
        attr_objs = atoms
    elif class_obj == Residue:
        attr_objs = atoms.residues
    else:
        attr_objs = atoms.structures
    from chimerax.core.commands import plural_of
    attr_names = plural_of(attr_name)
    needs_none_processing = True
    attr_vals = None
    if hasattr(attr_objs, attr_names):
        # attribute found in Collection; try to maximize efficiency
        needs_none_processing = False
        if average == 'residues' and class_obj == Atom:
            residues = atoms.unique_residues
            res_average = { r: getattr(r.atoms, attr_names).mean() for r in residues }
            attr_vals = [res_average[r] for r in atoms.residues]
        else:
            import numpy
            attr_vals = getattr(attr_objs, attr_names)
            if not isinstance(attr_vals, numpy.ndarray):
                # might have Nones
                needs_none_processing = True
        if not needs_none_processing:
            aradii = value_radii(way_points, attr_vals)
    if needs_none_processing:
        if attr_vals is None:
            attr_vals = [getattr(o, attr_name, None) for o in attr_objs]
        has_none = None in attr_vals
        if has_none:
            if average == 'residues' and class_obj == Atom:
                residues = atoms.unique_residues
                res_average = {}
                for r in residues:
                    vals = []
                    for a in r.atoms:
                        val = getattr(a, attr_name, None)
                        if val is not None:
                            vals.append(val)
                    res_average[r] = sum(vals)/len(vals) if vals else None
                attr_vals = [res_average[r] for r in atoms.residues]
            non_none_attr_vals = [v for v in attr_vals if v is not None]
            if non_none_attr_vals:
                non_none_radii = value_radii(way_points, non_none_attr_vals)
            else:
                non_none_radii = None
                session.logger.warning("All '%s' values are None" % attr_name)
            aradii = none_possible_radii(atoms.radii, attr_vals, non_none_radii, no_value_radius)
            # for later min/max message...
            attr_vals = non_none_attr_vals
        else:
            if average == 'residues' and class_obj == Atom:
                residues = atoms.unique_residues
                res_average = { r: sum([getattr(a, attr_name)
                    for a in r.atoms])/r.num_atoms for r in residues }
                attr_vals = [res_average[r] for r in atoms.residues]
            aradii = value_radii(way_points, attr_vals)
    if style != "unchanged":
        draw_mode = Atom.BALL_STYLE if style == "ball" else Atom.SPHERE_STYLE
        undo_state.add(atoms, "draw_modes", atoms.draw_modes, draw_mode)
        atoms.draw_modes = draw_mode
    undo_state.add(atoms, "radii", atoms.radii, aradii)
    session.undo.register(undo_state)
    atoms.radii = aradii
    if len(attr_vals):
        range_msg = 'atom %s range' if average is None else 'residue average %s range'
        msg = '%d atoms, %s %.3g to %.3g' % (
            len(atoms), (range_msg % attr_name), min(attr_vals), max(attr_vals))
        session.logger.status(msg, log=True)

# also used by cartoon byattribute
def none_possible_radii(item_radii, attr_vals, non_none_radii, no_value_radius):
    ri = 0
    radii = []
    import sys
    for item_radius, val in zip(item_radii, attr_vals):
        if val is None:
            if no_value_radius is None:
                radii.append(item_radius)
            else:
                radii.append(no_value_radius)
        else:
            radii.append(non_none_radii[ri])
            ri += 1
    import numpy
    return numpy.array(radii, numpy.single)

# also used by cartoon byattribute
def value_radii(way_points, values):
    from chimerax.surface.colorvol import _use_full_range, _colormap_with_range
    min_val, max_val = min(values), max(values)
    final_way_points = []
    for val, rad in way_points:
        if val == "min":
            wp = (min_val, rad)
        elif val == "max":
            wp = (max_val, rad)
        else:
            wp = (val, rad)
        final_way_points.append(wp)
    final_way_points.sort()
    # There will only likely be a few distinct radii values, so try to leverage that fact for efficiency
    import numpy
    radii = numpy.empty(len(values), numpy.single)
    # set() apparently does some fuzzy testing for whether floating-point values are equal, so...
    for val in numpy.unique(values):
        rad = _rad_lookup(val, final_way_points)
        radii[values == val] = rad
    return radii

def _rad_lookup(val, way_points):
    for i, wp in enumerate(way_points):
        right_val, right_rad = wp
        if val <= right_val:
            if i == 0:
                return right_rad
            left_val, left_rad = way_points[i-1]
            return left_rad + ((val - left_val) / (right_val - left_val)) * (right_rad - left_rad)
    return right_rad
# -----------------------------------------------------------------------------
# 

# also used by cartoon byattribute
class AttrRadiusPairArg(Annotation):
    name = "attr-value:radius pair"

    @staticmethod
    def parse(text, session):
        from chimerax.core.commands import AnnotationError, next_token
        if not text:
            raise AnnotationError("Expected %s" % AttrRadiusPairArg.name)
        token, text, rest = next_token(text)
        if ':' not in token:
            raise AnnotationError("No ':' found")

        attr_token, radius_token = token.split(':', 1)
        if not attr_token:
            raise AnnotationError("No attribute value before ':' in %s" % AttrRadiusPairArg.name)
        if not radius_token:
            raise AnnotationError("No radius value after ':' in %s" % AttrRadiusPairArg.name)
        if attr_token in ('min', 'max'):
            attr_val = attr_token
        else:
            attr_val, ignore, attr_rest = FloatArg.parse(attr_token, session)
            if attr_rest:
                raise AnnotationError("Trailing text after attribute value '%s'" % attr_token)
        radius, ignore, radius_rest = PositiveFloatArg.parse(radius_token, session)
        if radius_rest:
            raise AnnotationError("Trailing text after radius value '%s'" % radius_token)
        return (attr_val, radius), text, rest

    @staticmethod
    def unparse(value, session=None):
        attr_val, radius = value
        return "%g:%g" % (attr_val, radius)

#TODO: revise tool to use same default as command
def register_command(logger):
    desc = CmdDesc(required = [('objects', Or(ObjectsArg, EmptyArg))],
                   keyword = [('atom_radius', Or(EnumOf(['default']), FloatOrDeltaArg)),
                              ('stick_radius', FloatOrDeltaArg),
                              ('pseudobond_radius', FloatOrDeltaArg),
                              ('ball_scale', FloatOrDeltaArg)],
                   synopsis='change atom and bond sizes')
    register('size', desc, size, logger=logger)

    # size atoms by attribute
    from chimerax.atomic import AtomsArg
    from .size import AttrRadiusPairArg
    desc = CmdDesc(required=[('attr_name', StringArg),
                            ('atoms', Or(AtomsArg, EmptyArg))],
                   optional=[('way_points', RepeatOf(AttrRadiusPairArg))],
                   keyword=[('average', EnumOf(('residues',))),
                            ('no_value_radius', PositiveFloatArg),
                            ('style', AtomRadiiStyleArg)],
                   synopsis="size atoms by attribute value")
    register('size byattribute', desc, size_by_attr, logger=logger)

