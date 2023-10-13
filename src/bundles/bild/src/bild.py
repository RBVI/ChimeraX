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
bild: bild format support
=========================

Read a subset of Chimera's
`bild format <http://www.cgl.ucsf.edu/chimera/docs/UsersGuide/bild.html>`_:
.comment, .color, .transparency, .sphere, .cylinder, .arrow, .box,
.pop, .rotate, .scale, .translate.

The plan is to support all of the existing bild format.
"""

from chimerax.core.errors import UserError
from numpy import array, empty, float32, int32, uint8, array_equal
from chimerax.geometry import identity, translation, rotation, scale, distance, z_align
from chimerax.atomic import AtomicShapeDrawing, AtomicShapeInfo
from chimerax import surface


def _interp(t, a, b):
    return [a[i] + t * (b[i] - a[i]) for i in range(3)]


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
        self.session = session
        self.shapes = []
        # parse input
        self.warned = set()
        self.lineno = 0
        self.transforms = [identity()]
        self.pure = [True]   # True if corresponding transform has pure rotation
        self.cur_color = [1.0, 1.0, 1.0, 1.0]
        self.cur_transparency = 0
        self.cur_pos = array([0.0, 0.0, 0.0])
        self.cur_pos_is_move = True  # set whenever cur_pos is set
        self.cur_char_pos = array([0.0, 0.0, 0.0])
        self.cur_font = ['SANS', 12, 'PLAIN']
        self.cur_description = None
        self.cur_atoms = None
        self.num_objects = 0
        self.LINE_RADIUS = 0.08

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
                    self.session.logger.warning('text is not implemented on line %d' % self.lineno)
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
        drawing = AtomicShapeDrawing('shapes')
        self.model.add_drawing(drawing)
        if self.shapes:
            drawing.add_shapes(self.shapes)
        return [self.model], "Opened BILD data containing %d objects" % self.num_objects

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
            raise ValueError("Expected 'x1 y1 z1 x2 y2 z2 [r1 [r2 [rho]]]' after %s" % tokens[0])
        data = [self.parse_float(x) for x in tokens[1:]]
        r1 = data[6] if len(tokens) > 7 else 0.1
        r2 = data[7] if len(tokens) > 8 else 4 * r1
        rho = data[8] if len(tokens) > 9 else 0.75
        p1 = array(data[0:3])
        p2 = array(data[3:6])
        junction = p1 + rho * (p2 - p1)
        self.num_objects += 1
        if self.cur_description is not None:
            description = self.cur_description
        else:
            description = 'object %d: arrow' % self.num_objects
        vertices, normals, triangles = get_cylinder(
            r1, p1, junction,
            closed=True, xform=self.transforms[-1], pure=self.pure[-1])
        vertices2, normals2, triangles2 = get_cone(
            r2, junction, p2, bottom=True,
            xform=self.transforms[-1], pure=self.pure[-1])
        vertices, normals, triangles = combine_triangles((
            (vertices, normals, triangles),
            (vertices2, normals2, triangles2)
        ))
        if vertices is None:
            raise ValueError("degenerate arrow")
        else:
            shape = AtomicShapeInfo(
                vertices, normals, triangles,
                _cvt_color(self.cur_color), self.cur_atoms, description)
            self.shapes.append(shape)

    def associate_command(self, tokens):
        atomspec = ' '.join(tokens[1:])
        if not atomspec:
            self.cur_atoms = None
            return
        from chimerax.core.commands import AtomSpecArg
        a, _, _ = AtomSpecArg.parse(atomspec, self.session)
        self.cur_atoms = a.evaluate(self.session).atoms

    def box_command(self, tokens):
        if len(tokens) != 7:
            raise ValueError("Expected 'x1 y1 z1 x2 y2 z2' after %s" % tokens[0])
        data = [self.parse_float(x) for x in tokens[1:7]]
        llb = array(data[0:3])
        urf = array(data[3:6])
        self.num_objects += 1
        if self.cur_description is not None:
            description = self.cur_description
        else:
            description = 'object %d: box' % self.num_objects
        vertices, normals, triangles = get_box(llb, urf, self.transforms[-1], pure=self.pure[-1])
        if vertices is None:
            raise ValueError("degenerate box")
        else:
            shape = AtomicShapeInfo(
                vertices, normals, triangles,
                _cvt_color(self.cur_color), self.cur_atoms, description)
            self.shapes.append(shape)

    def cmov_command(self, tokens):
        if len(tokens) != 4:
            raise UserError("Expected 'x y z' after %s" % tokens[0])
        data = [self.parse_float(x) for x in tokens[1:4]]
        xyz = array(data[0:3])
        self.cur_char_pos = xyz

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
            raise ValueError("Expected 'R G B' values or color name after %s" % tokens[0])
        else:
            self.cur_color[0:3] = [self.parse_float(x) for x in tokens[1:4]]

    def cone_command(self, tokens):
        if len(tokens) not in (8, 9) or (
                len(tokens) == 9 and tokens[8] != 'open'):
            raise ValueError("Expected 'x1 y1 z1 x2 y2 z2 radius [open]' after %s" % tokens[0])
        data = [self.parse_float(x) for x in tokens[1:8]]
        p0 = array(data[0:3])
        p1 = array(data[3:6])
        radius = data[6]
        if len(tokens) < 9:
            bottom = True
        else:
            bottom = False
        self.num_objects += 1
        if self.cur_description is not None:
            description = self.cur_description
        else:
            description = 'object %d: cone' % self.num_objects
        vertices, normals, triangles = get_cone(
            radius, p0, p1, bottom=bottom, xform=self.transforms[-1], pure=self.pure[-1])
        if vertices is None:
            raise ValueError("degenerate cone")
        else:
            shape = AtomicShapeInfo(
                vertices, normals, triangles,
                _cvt_color(self.cur_color), self.cur_atoms, description)
            self.shapes.append(shape)

    def cylinder_command(self, tokens):
        if len(tokens) not in (8, 9) or (
                len(tokens) == 9 and tokens[8] != 'open'):
            raise ValueError("Expected 'x1 y1 z1 x2 y2 z2 radius [open]' after %s" % tokens[0])
        data = [self.parse_float(x) for x in tokens[1:8]]
        p0 = array(data[0:3])
        p1 = array(data[3:6])
        radius = data[6]
        if len(tokens) < 9:
            closed = True
        else:
            closed = False
        self.num_objects += 1
        if self.cur_description is not None:
            description = self.cur_description
        else:
            description = 'object %d: cylinder' % self.num_objects
        vertices, normals, triangles = get_cylinder(
            radius, p0, p1, closed=closed, xform=self.transforms[-1], pure=self.pure[-1])
        if vertices is None:
            raise ValueError("degenerate cylinder")
        else:
            shape = AtomicShapeInfo(
                vertices, normals, triangles,
                _cvt_color(self.cur_color), self.cur_atoms, description)
            self.shapes.append(shape)

    def dashed_cylinder_command(self, tokens):
        if len(tokens) not in (9, 10) or (
                len(tokens) == 10 and tokens[9] != 'open'):
            raise ValueError("Expected 'count x1 y1 z1 x2 y2 z2 radius [open]' after %s" % tokens[0])
        count = self.parse_int(tokens[1])
        data = [self.parse_float(x) for x in tokens[2:9]]
        p0 = array(data[0:3])
        p1 = array(data[3:6])
        radius = data[6]
        if len(tokens) < 10:
            closed = True
        else:
            closed = False
        self.num_objects += 1
        if self.cur_description is not None:
            description = self.cur_description
        else:
            description = 'object %d: dashed cylinder' % self.num_objects
        vertices, normals, triangles = get_dashed_cylinder(
            count, radius, p0, p1, closed=closed, xform=self.transforms[-1], pure=self.pure[-1])
        if vertices is None:
            raise ValueError("degenerate dashed cylinder")
        else:
            shape = AtomicShapeInfo(
                vertices, normals, triangles,
                _cvt_color(self.cur_color), self.cur_atoms, description)
            self.shapes.append(shape)

    def dot_command(self, tokens):
        if len(tokens) != 4:
            raise UserError("Expected 'x y z' after %s" % tokens[0])
        data = [self.parse_float(x) for x in tokens[1:4]]
        center = array(data[0:3])
        radius = 1
        self.num_objects += 1
        if self.cur_description is not None:
            description = self.cur_description
        else:
            description = 'object %d: dot' % self.num_objects
        vertices, normals, triangles = get_sphere(
            radius, center, self.transforms[-1], pure=self.pure[-1])
        shape = AtomicShapeInfo(
            vertices, normals, triangles,
            _cvt_color(self.cur_color), self.cur_atoms, description)
        self.shapes.append(shape)
        self.cur_pos = center
        self.cur_pos_is_move = False

    def draw_command(self, tokens):
        if len(tokens) != 4:
            raise ValueError("Expected 'x y z' after %s" % tokens[0])
        data = [self.parse_float(x) for x in tokens[1:4]]
        xyz = array(data[0:3])
        radius = self.LINE_RADIUS
        p0 = self.cur_pos
        if tokens[0] in ('.draw', '.d'):
            p1 = xyz
        else:
            p1 = p0 + xyz
        self.num_objects += 1
        if self.cur_description is not None:
            description = self.cur_description
        else:
            description = 'object %d: vector' % self.num_objects
        vertices, normals, triangles = get_sphere(
            radius, p1, self.transforms[-1], pure=self.pure[-1])
        vertices2, normals2, triangles2 = get_cylinder(
            radius, p0, p1, closed=False, xform=self.transforms[-1], pure=self.pure[-1])
        if self.cur_pos_is_move:
            vertices3, normals3, triangles3 = get_sphere(
                radius, p0, self.transforms[-1], pure=self.pure[-1])
            vertices, normals, triangles = combine_triangles((
                (vertices, normals, triangles),
                (vertices2, normals2, triangles2),
                (vertices3, normals3, triangles3)
            ))
        else:
            vertices, normals, triangles = combine_triangles((
                    (vertices, normals, triangles),
                    (vertices2, normals2, triangles2)
            ))
        shape = AtomicShapeInfo(
            vertices, normals, triangles,
            _cvt_color(self.cur_color), self.cur_atoms, description)
        self.shapes.append(shape)
        self.cur_pos = p1
        self.cur_pos_is_move = False

    def marker_command(self, tokens):
        if len(tokens) != 4:
            raise ValueError("Expected 'x y z' after %s" % tokens[0])
        data = [self.parse_float(x) for x in tokens[1:4]]
        center = array(data[0:3])
        self.num_objects += 1
        if self.cur_description is not None:
            description = self.cur_description
        else:
            description = 'object %d: marker' % self.num_objects
        llb = center - 0.5
        urf = center + 0.5
        vertices, normals, triangles = get_box(llb, urf, self.transforms[-1], pure=self.pure[-1])
        shape = AtomicShapeInfo(
            vertices, normals, triangles,
            _cvt_color(self.cur_color), self.cur_atoms, description)
        self.shapes.append(shape)
        self.cur_pos = center
        self.cur_pos_is_move = False

    def font_command(self, tokens):
        # TODO: need to handle platform font families with spaces in them
        if len(tokens) not in (3, 4):
            raise UserError("Expected 'fontFamily pointSize [style]' after %s" % tokens[0])
        family = tokens[1].lower()
        if family in ('times', 'serif'):
            family = 'SERIF'
        elif family in ('helvetica', 'sans'):
            family = 'SANS'
        elif family in ('courier', 'typewriter'):
            family = 'TYPEWRITER'
        else:
            raise UserError('Unknown font family')
        size = self.parse_int(tokens[2])
        if size < 1:
            raise UserError('Font size must be at least 1')
        if len(tokens) == 3:
            style = 'PLAIN'
        else:
            style = tokens[3].lower()
            if style not in ('plain', 'bold', 'italic', 'bolditalic'):
                raise UserError('unknown font style')
            style = style.upper()
        self.cur_font = [family, size, style]

    def move_command(self, tokens):
        if len(tokens) != 4:
            raise UserError("Expected 'x y z' after %s" % tokens[0])
        data = [self.parse_float(x) for x in tokens[1:4]]
        xyz = array(data[0:3])
        if tokens[0] in ('.move', '.m'):
            self.cur_pos = xyz
        else:
            self.cur_pos += xyz
        self.cur_pos_is_move = True

    def note_command(self, tokens):
        description = ' '.join(tokens[1:])
        if not description:
            self.cur_description = None
        else:
            self.cur_description = description

    def polygon_command(self, tokens):
        # TODO: use GLU to tesselate polygon
        #     for now, find center and make a triangle fan
        if len(tokens) % 3 != 1:
            raise UserError("Expected 'x1 y1 z1 ... xN yN zN' after %s" % tokens[0])
        data = [self.parse_float(x) for x in tokens[1:]]
        vertices = array(data, dtype=float32)
        n = len(data) // 3
        vertices.shape = (n, 3)
        if n < 3:
            raise UserError("Need at least 3 vertices in a polygon")
        self.num_objects += 1
        if self.cur_description is not None:
            description = self.cur_description
        else:
            description = 'object %d: polygon' % self.num_objects
        from chimerax.geometry import Plane
        plane = Plane(vertices)
        loops = ((0, len(vertices) - 1),)
        t = surface.triangulate_polygon(loops, plane.normal, vertices)
        normals = empty(vertices.shape, dtype=float32)
        normals[:] = plane.normal
        triangles = array(t, dtype=int32)
        shape = AtomicShapeInfo(
            vertices, normals, triangles,
            _cvt_color(self.cur_color), self.cur_atoms, description)
        self.shapes.append(shape)

    def pop_command(self, tokens):
        if len(self.transforms) == 1:
            raise ValueError("Empty transformation stack")
        self.transforms.pop()
        self.pure.pop()

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
                raise UserError("Expected 'x', 'y', or 'z' axis in %s" % tokens[0])
        else:
            data = [self.parse_float(x) for x in tokens[1:5]]
            angle = data[0]
            axis = array(data[1:4])
        xform = rotation(axis, angle)
        self.transforms.append(self.transforms[-1] * xform)
        self.pure.append(self.pure[-1])

    def scale_command(self, tokens):
        if len(tokens) not in (2, 3, 4):
            raise ValueError("Expected 'x [y [z]]' after %s" % tokens[0])
        data = [self.parse_float(x) for x in tokens[1:]]
        if len(data) == 1:
            data.extend([data[0], data[0]])
        elif len(data) == 2:
            data.append(data[0])
        xform = scale(data)
        self.transforms.append(self.transforms[-1] * xform)
        self.pure.append(False)

    def sphere_command(self, tokens):
        if len(tokens) != 5:
            raise UserError("Expected 'x y z radius' after %s" % tokens[0])
        data = [self.parse_float(x) for x in tokens[1:5]]
        center = array(data[0:3])
        radius = data[3]
        self.num_objects += 1
        if self.cur_description is not None:
            description = self.cur_description
        else:
            description = 'object %d: sphere' % self.num_objects
        vertices, normals, triangles = get_sphere(
            radius, center, self.transforms[-1], pure=self.pure[-1])
        if vertices is None:
            raise ValueError("degenerate sphere")
        else:
            shape = AtomicShapeInfo(
                vertices, normals, triangles,
                _cvt_color(self.cur_color), self.cur_atoms, description)
            self.shapes.append(shape)

    def translate_command(self, tokens):
        if len(tokens) != 4:
            raise ValueError("Expected 'x y z' after %s" % tokens[0])
        data = [self.parse_float(x) for x in tokens[1:4]]
        xform = translation(data)
        self.transforms.append(self.transforms[-1] * xform)
        self.pure.append(self.pure[-1])

    def transparency_command(self, tokens):
        if len(tokens) != 2:
            raise UserError("Expected 'value' after %s" % tokens[0])
        self.cur_transparency = self.parse_float(tokens[1])
        self.cur_color[3] = 1 - self.cur_transparency

    def vector_command(self, tokens):
        if len(tokens) != 7:
            raise ValueError("Expected 'x1 y1 z1 x2 y2 z2' after %s" % tokens[0])
        data = [self.parse_float(x) for x in tokens[1:7]]
        p0 = array(data[0:3])
        p1 = array(data[3:6])
        radius = self.LINE_RADIUS
        self.num_objects += 1
        if self.cur_description is not None:
            description = self.cur_description
        else:
            description = 'object %d: vector' % self.num_objects
        vertices, normals, triangles = get_sphere(
            radius, p0, self.transforms[-1], pure=self.pure[-1])
        vertices2, normals2, triangles2 = get_cylinder(
            radius, p0, p1, closed=False, xform=self.transforms[-1], pure=self.pure[-1])
        vertices3, normals3, triangles3 = get_sphere(
            radius, p1, self.transforms[-1], pure=self.pure[-1])
        vertices, normals, triangles = combine_triangles((
            (vertices, normals, triangles),
            (vertices2, normals2, triangles2),
            (vertices3, normals3, triangles3)
        ))
        shape = AtomicShapeInfo(
            vertices, normals, triangles,
            _cvt_color(self.cur_color), self.cur_atoms, description)
        self.shapes.append(shape)
        self.cur_pos = p1
        self.cur_pos_is_move = False

    _commands = {
        '.arrow': arrow_command,
        '.associate': associate_command,
        '.box': box_command,
        '.c': comment_command,
        '.cmov': cmov_command,
        '.comment': comment_command,
        '.color': color_command,
        '.cone': cone_command,
        '.cylinder': cylinder_command,
        '.d': draw_command,
        '.dashedcylinder': dashed_cylinder_command,
        '.dot': dot_command,
        '.dotat': dot_command,
        '.dr': draw_command,
        '.draw': draw_command,
        '.drawrel': draw_command,
        '.font': font_command,
        '.m': move_command,
        '.marker': marker_command,
        '.move': move_command,
        '.mr': move_command,
        '.moverel': move_command,
        '.note': note_command,
        '.polygon': polygon_command,
        '.pop': pop_command,
        '.rot': rotate_command,
        '.rotate': rotate_command,
        '.scale': scale_command,
        '.sphere': sphere_command,
        '.tran': translate_command,
        '.translate': translate_command,
        '.transparency': transparency_command,
        '.v': vector_command,
        '.vector': vector_command,
    }


def read_bild(session, stream, file_name):
    """Populate the scene with the geometry from a bild file

    :param stream: either a binary I/O stream or the name of a file

    Extra arguments are ignored.
    """
    b = _BildFile(session, file_name)
    return b.parse_stream(stream)


def _cvt_color(color):
    color = (array([color]) * 255).astype(uint8)
    return color


def get_sphere(radius, center, xform=None, pure=False):
    # TODO: vary number of triangles with radius
    if radius == 0:
        return None, None, None
    vertices, normals, triangles = surface.sphere_geometry2(200)
    vertices = vertices * radius + center
    if xform is not None:
        xform.transform_points(vertices, in_place=True)
        xform.transform_normals(normals, in_place=True, is_rotation=pure)
    return vertices, normals, triangles


def get_cylinder(radius, p0, p1, closed=True, xform=None, pure=False):
    h = distance(p0, p1)
    if h == 0:
        return None, None, None
    vertices, normals, triangles = surface.cylinder_geometry(radius, height=h, caps=closed)
    # rotate so z-axis matches p0->p1
    xf = z_align(p0, p1)
    inverse = xf.inverse()
    vertices = inverse * (vertices + [0, 0, h / 2])
    inverse.transform_normals(normals, in_place=True, is_rotation=True)
    if xform is not None:
        xform.transform_points(vertices, in_place=True)
        xform.transform_normals(normals, in_place=True, is_rotation=pure)
    return vertices, normals, triangles


def get_dashed_cylinder(count, radius, p0, p1, closed=True, xform=None, pure=False):
    h = distance(p0, p1)
    if h == 0:
        return None, None, None
    vertices, normals, triangles = surface.dashed_cylinder_geometry(count, radius, height=h, caps=closed)
    # rotate so z-axis matches p0->p1
    from chimerax.geometry import z_align
    xf = z_align(p0, p1)
    inverse = xf.inverse()
    vertices = inverse * (vertices + [0, 0, h / 2])
    inverse.transform_normals(normals, in_place=True, is_rotation=True)
    if xform is not None:
        xform.transform_points(vertices, in_place=True)
        xform.transform_normals(normals, in_place=True, is_rotation=pure)
    return vertices, normals, triangles


def get_box(llb, urf, xform=None, pure=False):
    if array_equal(llb, urf):
        return None, None, None
    vertices, normals, triangles = surface.box_geometry(llb, urf)
    if xform is not None:
        xform.transform_points(vertices, in_place=True)
        xform.transform_normals(normals, in_place=True, is_rotation=pure)
    return vertices, normals, triangles


def get_cone(radius, p0, p1, bottom=False, xform=None, pure=False):
    h = distance(p0, p1)
    if h == 0:
        return None, None, None
    vertices, normals, triangles = surface.cone_geometry(radius, height=h, caps=bottom)
    from chimerax.geometry import z_align
    xf = z_align(p0, p1)
    inverse = xf.inverse()
    vertices = inverse * (vertices + [0, 0, h / 2])
    inverse.transform_normals(normals, in_place=True, is_rotation=True)
    if xform is not None:
        xform.transform_points(vertices, in_place=True)
        xform.transform_normals(normals, in_place=True, is_rotation=pure)
    return vertices, normals, triangles


def combine_triangles(triangle_info):
    from numpy import empty, float32, int32, concatenate as concat
    all_vertices = []
    all_normals = []
    all_triangles = []
    num_vertices = 0
    num_triangles = 0
    for i, info in enumerate(triangle_info):
        vertices, normals, triangles = info
        if vertices is None:
            continue
        all_vertices.append(vertices)
        all_normals.append(normals)
        all_triangles.append(triangles + num_vertices)
        num_vertices += len(vertices)
        num_triangles += len(triangles)
    if len(all_vertices) == 0:
        return None, None, None
    vertices = empty((num_vertices, 3), dtype=float32)
    normals = empty((num_vertices, 3), dtype=float32)
    triangles = empty((num_triangles, 3), dtype=int32)
    return (
        concat(all_vertices, out=vertices),
        concat(all_normals, out=normals),
        concat(all_triangles, out=triangles)
    )
