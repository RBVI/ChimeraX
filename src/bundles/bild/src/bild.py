# vim: set expandtab shiftwidth=4 softtabstop=4:

# === UCSF ChimeraX Copyright ===
# Copyright 2017 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

"""
bild: bild format support
=========================

Read a subset of Chimera's
`bild format <http://www.cgl.ucsf.edu/chimera/docs/UsersGuide/bild.html>`_:
.comment, .color, .transparency, .sphere, .cylinder, .arrow, .box,
.pop, .rotate, .scale, .translate.

The plan is to support all of the existing bild format.
"""

from chimerax.core.errors import UserError
import numpy
from chimerax.core.geometry import identity, translation, rotation, scale, distance, z_align, normalize_vectors
from chimerax.core import surface
from .drawing import ShapeDrawing


def _interp(t, a, b):
    return [t * (b[i] - a[i]) for i in range(3)]


def _rgb_color(color_number):
    # backwards compatible Midas colors
    if color_number == 0:
        return (1, 1, 1)
    if color_number < 9:
        return _interp((color_number - 1) / 8.0, (0, 1, 0), (0, 1, 1))
    if color_number < 17:
        return _interp((color_number - 8) / 8.0, (0, 1, 1), (0, 0, 1))
    if color_number < 25:
        return _interp((color_number - 16) / 8.0, (0, 0, 1), (1, 0, 1))
    if color_number < 33:
        return _interp((color_number - 24) / 8.0, (1, 0, 1), (1, 0, 0))
    if color_number < 49:
        return _interp((color_number - 32) / 16.0, (1, 0, 0), (1, 1, 0))
    if color_number < 65:
        return _interp((color_number - 48) / 16.0, (1, 1, 0), (0, 0, 0))
    if color_number == 65:
        return (0.7, 0.7, 0.7)
    raise ValueError("Color number must be from 0 to 65 inclusive")


def _is_int(i):
    try:
        int(i)
        return True
    except ValueError:
        return False


class _BildFile:

    def __init__(self, session, filename):
        from chimerax.core import generic3d
        self.model = generic3d.Generic3DModel(filename, session)
        self.drawing = ShapeDrawing('shapes')
        self.model.add_drawing(self.drawing)
        self.session = session
        # parse input
        self.warned = set()
        self.lineno = 0
        self.transforms = [identity()]
        self.cur_color = [1.0, 1.0, 1.0, 1.0]
        self.cur_transparency = 0
        self.cur_atoms = None
        self.num_objects = 0
        self.num_arrows = 0
        self.num_boxes = 0
        self.num_cylinders = 0
        self.num_cones = 0
        self.num_spheres = 0

    def parse_stream(self, stream):
        for line in stream.readlines():
            self.lineno += 1
            line = line.decode('utf-8', 'ignore').rstrip()
            tokens = line.split()
            if not tokens:
                # ignore empty line
                continue
            if line[0] != '.':
                # TODO: text
                if 'text' not in self.warned:
                    self.warned.add('text')
                    self.session.logger.warning('text is not supported on line %d' % self.lineno)
                continue
            func = self._commands.get(tokens[0], None)
            if func is None:
                if tokens[0] not in self.warned:
                    self.warned.add(tokens[0])
                    self.session.logger.warning(
                        'Unknown command %s on line %d' % (tokens[0], self.lineno))
                continue
            try:
                func(self, tokens)
            except ValueError as e:
                self.session.logger.warning('%s on line %d' % (e, self.lineno))
        return [self.model], "Opened BILD data containing %d objects" % self.num_objects

    def unimplemented(self, command, tokens):
        if command in self.warned:
            return
        self.warned.add(command)
        self.session.logger.warning(
            '%s command is not implemented on line %d' % (command, self.lineno))

    def parse_color(self, x):
        # Use *Arg for consistent error messages
        from chimerax.core.commands import ColorArg
        return ColorArg.parse(x, self.session)[0]

    def parse_int(self, x):
        # Use *Arg for consistent error messages
        from chimerax.core.commands import IntArg
        return IntArg.parse(x, self.session)[0]

    def parse_float(self, x):
        # Use *Arg for consistent error messages
        from chimerax.core.commands import FloatArg
        return FloatArg.parse(x, self.session)[0]

    def arrow_command(self, tokens):
        if len(tokens) not in (7, 8, 9, 10):
            raise ValueError("Expected 'x1 y1 z1 x2 y2 z2 [r1 [r2 [rho]]]' after .arrow")
        data = [self.parse_float(x) for x in tokens[1:]]
        r1 = data[6] if len(tokens) > 7 else 0.1
        r2 = data[7] if len(tokens) > 8 else 4 * r1
        rho = data[8] if len(tokens) > 9 else 0.75
        p1 = numpy.array(data[0:3])
        p2 = numpy.array(data[3:6])
        junction = p1 + rho * (p2 - p1)
        self.num_arrows += 1
        balloon_text = 'arrow %d' % self.num_arrows
        vertices, normals, triangles = get_cylinder(
            r1, p1, junction,
            closed=True, xform=self.transforms[-1])
        v, n, t = get_cone(r2, junction, p2, bottom=True,
                           xform=self.transforms[-1])
        from numpy import concatenate as concat
        t += len(vertices)
        self.drawing.add_shape(
            concat((vertices, v)), concat((normals, n)), concat((triangles, t)),
            _cvt_color(self.cur_color), self.cur_atoms, balloon_text)
        self.num_objects += 1

    def atomspec_command(self, tokens):
        atomspec = ' '.join(tokens[1:])
        if not atomspec or atomspec == 'none':
            atomspec = None
        from chimerax.core.commands import AtomSpecArg
        a, _, _ = AtomSpecArg(atomspec)
        self.cur_atoms = a.evaluate(self.session).atoms

    def box_command(self, tokens):
        if len(tokens) != 7:
            raise ValueError("Expected 'x1 y1 z1 x2 y2 z2' after .box")
        data = [self.parse_float(x) for x in tokens[1:7]]
        llb = numpy.array(data[0:3])
        urf = numpy.array(data[3:6])
        self.num_boxes += 1
        balloon_text = 'box %d' % self.num_boxes
        vertices, normals, triangles = get_box(llb, urf, self.transforms[-1])
        self.drawing.add_shape(
            vertices, normals, triangles,
            _cvt_color(self.cur_color), self.cur_atoms, balloon_text)
        self.num_objects += 1

    def comment_command(self, tokens):
        # ignore comments
        pass

    def color_command(self, tokens):
        if len(tokens) == 2:
            if _is_int(tokens[1]):
                self.cur_color[0:3] = _rgb_color(self.parse_int(tokens[1]))
            else:
                c = self.parse_color(tokens[1])
                if hasattr(c, 'explicit_transparency') and c.explicit_transparency:
                    self.cur_color[0:4] = c.rgba
                else:
                    self.cur_color[0:3] = c.rgba[0:3]
                    self.cur_color[3] = 1 - self.cur_transparency
        elif len(tokens) != 4:
            raise ValueError("Expected 'R G B' values or color name after .color")
        else:
            self.cur_color[0:3] = [self.parse_float(x) for x in tokens[1:4]]

    def cone_command(self, tokens):
        if len(tokens) not in (8, 9) or (
                len(tokens) == 9 and tokens[8] != 'open'):
            raise ValueError("Expected 'x1 y1 z1 x2 y2 z2 radius [open]' after .cylinder")
        data = [self.parse_float(x) for x in tokens[1:8]]
        p0 = numpy.array(data[0:3])
        p1 = numpy.array(data[3:6])
        radius = data[6]
        if len(tokens) < 9:
            bottom = True
        else:
            bottom = False
        self.num_cones += 1
        balloon_text = 'cone %d' % self.num_cones
        vertices, normals, triangles = get_cone(
            radius, p0, p1, bottom=bottom, xform=self.transforms[-1])
        self.drawing.add_shape(
            vertices, normals, triangles,
            _cvt_color(self.cur_color), self.cur_atoms, balloon_text)
        self.num_objects += 1

    def cylinder_command(self, tokens):
        if len(tokens) not in (8, 9) or (
                len(tokens) == 9 and tokens[8] != 'open'):
            raise ValueError("Expected 'x1 y1 z1 x2 y2 z2 radius [open]' after .cylinder")
        data = [self.parse_float(x) for x in tokens[1:8]]
        p0 = numpy.array(data[0:3])
        p1 = numpy.array(data[3:6])
        radius = data[6]
        if len(tokens) < 9:
            closed = True
        else:
            closed = False
        self.num_cylinders += 1
        balloon_text = 'cylinder %d' % self.num_cylinders
        vertices, normals, triangles = get_cylinder(
            radius, p0, p1, closed=closed, xform=self.transforms[-1])
        self.drawing.add_shape(
            vertices, normals, triangles,
            _cvt_color(self.cur_color), self.cur_atoms, balloon_text)
        self.num_objects += 1

    def dashed_cylinder_command(self, tokens):
        if len(tokens) not in (9, 10) or (
                len(tokens) == 10 and tokens[9] != 'open'):
            raise ValueError("Expected 'count x1 y1 z1 x2 y2 z2 radius [open]' after .cylinder")
        count = self.parse_int(tokens[1])
        data = [self.parse_float(x) for x in tokens[2:9]]
        p0 = numpy.array(data[0:3])
        p1 = numpy.array(data[3:6])
        radius = data[6]
        if len(tokens) < 10:
            closed = True
        else:
            closed = False
        self.num_cylinders += 1
        balloon_text = 'cylinder %d' % self.num_cylinders
        vertices, normals, triangles = get_dashed_cylinder(
            count, radius, p0, p1, closed=closed, xform=self.transforms[-1])
        self.drawing.add_shape(
            vertices, normals, triangles,
            _cvt_color(self.cur_color), self.cur_atoms, balloon_text)
        self.num_objects += 1

    def pop_command(self, tokens):
        if len(self.transforms) == 1:
            raise ValueError("Empty transformation stack")
        self.transforms.pop()

    def rotate_command(self, tokens):
        if len(tokens) not in (3, 5):
            raise ValueError("Expected 'angle axis' after %s" % tokens[0])
        if len(tokens) == 3:
            angle = self.parse_float(tokens[1])
            if tokens[2] == 'x':
                axis = (1., 0., 0.)
            elif tokens[2] == 'y':
                axis = (1., 0., 0.)
            elif tokens[2] == 'z':
                axis = (1., 0., 0.)
            else:
                raise UserError("Expected 'x', 'y', or 'z' axis")
        else:
            data = [self.parse_float(x) for x in tokens[1:5]]
            angle = data[0]
            axis = numpy.array(data[1:4])
        xform = rotation(axis, angle)
        self.transforms.append(self.transforms[-1] * xform)

    def scale_command(self, tokens):
        if len(tokens) not in (2, 3, 4):
            raise ValueError("Expected 'x [y [z]]' after .scale")
        data = [self.parse_float(x) for x in tokens[1:]]
        if len(data) == 1:
            data.extend([data[0], data[0]])
        elif len(data) == 2:
            data.append(data[0])
        xform = scale(data)
        self.transforms.append(self.transforms[-1] * xform)

    def sphere_command(self, tokens):
        if len(tokens) != 5:
            raise UserError("Expected 'x y z radius' after .sphere")
        data = [self.parse_float(x) for x in tokens[1:5]]
        center = numpy.array(data[0:3])
        radius = data[3]
        self.num_spheres += 1
        balloon_text = 'sphere %d' % self.num_spheres
        vertices, normals, triangles = get_sphere(
            radius, center, self.transforms[-1])
        self.drawing.add_shape(
            vertices, normals, triangles,
            _cvt_color(self.cur_color), self.cur_atoms, balloon_text)
        self.num_objects += 1

    def translate_command(self, tokens):
        if len(tokens) != 4:
            raise ValueError("Expected 'x y z' after %s" % tokens[0])
        data = [self.parse_float(x) for x in tokens[1:4]]
        xform = translation(data)
        self.transforms.append(self.transforms[-1] * xform)

    def transparency_command(self, tokens):
        if len(tokens) != 2:
            raise UserError("Expected 'value' after .transparency")
        self.cur_transparency = self.parse_float(tokens[1])
        self.cur_color[3] = 1 - self.cur_transparency

    _commands = {
        '.arrow': arrow_command,
        '.atomspec': atomspec_command,
        '.box': box_command,
        '.c': comment_command,
        '.cmov': lambda self, tokens: self.unimplemented('.cmov', tokens),
        '.comment': comment_command,
        '.color': color_command,
        '.cone': cone_command,
        '.cylinder': cylinder_command,
        '.d': lambda self, tokens: self.unimplemented('.d', tokens),
        '.dot': lambda self, tokens: self.unimplemented('.dot', tokens),
        '.dotat': lambda self, tokens: self.unimplemented('.dotat', tokens),
        '.dr': lambda self, tokens: self.unimplemented('.dr', tokens),
        '.draw': lambda self, tokens: self.unimplemented('.draw', tokens),
        '.drawrel': lambda self, tokens: self.unimplemented('.drawrel', tokens),
        '.font': lambda self, tokens: self.unimplemented('.font', tokens),
        '.m': lambda self, tokens: self.unimplemented('.m', tokens),
        '.marker': lambda self, tokens: self.unimplemented('.marker', tokens),
        '.move': lambda self, tokens: self.unimplemented('.move', tokens),
        '.mr': lambda self, tokens: self.unimplemented('.mr', tokens),
        '.moverel': lambda self, tokens: self.unimplemented('.moverel', tokens),
        '.polygon': lambda self, tokens: self.unimplemented('.polygon', tokens),
        '.pop': pop_command,
        '.rot': rotate_command,
        '.rotate': rotate_command,
        '.scale': scale_command,
        '.sphere': sphere_command,
        '.tran': translate_command,
        '.translate': translate_command,
        '.transparency': transparency_command,
        '.v': lambda self, tokens: self.unimplemented('.v', tokens),
        '.vector': lambda self, tokens: self.unimplemented('.vector', tokens),
    }


def read_bild(session, stream, file_name):
    """Populate the scene with the geometry from a bild file

    :param stream: either a binary I/O stream or the name of a file

    Extra arguments are ignored.
    """
    b = _BildFile(session, file_name)
    return b.parse_stream(stream)


def _cvt_color(color):
    color = (numpy.array([color]) * 255).astype(numpy.uint8)
    return color


def get_sphere(radius, center, xform=None):
    # TODO: vary number of triangles with radius
    vertices, normals, triangles = surface.sphere_geometry2(200)
    vertices = vertices * radius + center
    if xform is not None:
        vertices = xform * vertices
        normals = xform.apply_without_translation(normals)
        normalize_vectors(normals)
    return vertices, normals, triangles


def get_cylinder(radius, p0, p1, closed=True, xform=None):
    h = distance(p0, p1)
    vertices, normals, triangles = surface.cylinder_geometry(radius, height=h, caps=closed)
    # rotate so z-axis matches p0->p1
    xf = z_align(p0, p1)
    inverse = xf.inverse()
    vertices = inverse * (vertices + [0, 0, h / 2])
    normals = inverse.apply_without_translation(normals)
    if xform is not None:
        vertices = xform * vertices
        normals = xform.apply_without_translation(normals)
        normalize_vectors(normals)
    return vertices, normals, triangles


def get_dashed_cylinder(count, radius, p0, p1, closed=True, xform=None):
    h = distance(p0, p1)
    vertices, normals, triangles = surface.dashed_cylinder_geometry(count, radius, height=h, caps=closed)
    # rotate so z-axis matches p0->p1
    from chimerax.core.geometry import z_align
    xf = z_align(p0, p1)
    inverse = xf.inverse()
    vertices = inverse * (vertices + [0, 0, h / 2])
    normals = inverse.apply_without_translation(normals)
    if xform is not None:
        vertices = xform * vertices
        normals = xform.apply_without_translation(normals)
        normalize_vectors(normals)
    return vertices, normals, triangles


def get_box(llb, urf, xform=None):
    vertices, normals, triangles = surface.box_geometry(llb, urf)
    if xform is not None:
        vertices = xform * vertices
        normals = xform.apply_without_translation(normals)
        normalize_vectors(normals)
    return vertices, normals, triangles


def get_cone(radius, p0, p1, bottom=False, xform=None):
    h = distance(p0, p1)
    vertices, normals, triangles = surface.cone_geometry(radius, height=h, caps=bottom)
    from chimerax.core.geometry import z_align
    xf = z_align(p0, p1)
    inverse = xf.inverse()
    vertices = inverse * vertices
    normals = inverse.apply_without_translation(normals)
    if xform is not None:
        vertices = xform * vertices
        normals = xform.apply_without_translation(normals)
        normalize_vectors(normals)
    return vertices, normals, triangles
