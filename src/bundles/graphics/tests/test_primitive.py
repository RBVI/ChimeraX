import numpy
import pytest

from chimerax.graphics import (CylinderPrimitiveBatch, Drawing,
                               ExportGeometryContext, SpherePrimitiveBatch)


def test_primitive_validation_rejects_mismatched_arrays_and_negative_radius():
    colors = numpy.array(((255, 255, 255, 255),), numpy.uint8)
    with pytest.raises(ValueError):
        SpherePrimitiveBatch(numpy.zeros((1, 3), numpy.float32),
                             numpy.array((-1,), numpy.float32), colors)
    with pytest.raises(ValueError):
        CylinderPrimitiveBatch(numpy.zeros((1, 3), numpy.float32),
                               numpy.ones((1, 3), numpy.float32),
                               numpy.ones((2,), numpy.float32), colors)


def test_semantic_drawing_has_no_retained_triangles():
    spheres = SpherePrimitiveBatch(
        numpy.array(((1, 2, 3),), numpy.float32),
        numpy.array((2,), numpy.float32),
        numpy.array(((10, 20, 30, 255),), numpy.uint8))
    drawing = Drawing('semantic spheres')
    drawing.set_primitive_batch(spheres)

    assert drawing.vertices is None
    assert drawing.triangles is None
    assert drawing.has_export_geometry()
    assert not drawing.empty_drawing()


def test_sphere_export_is_transient_and_preserves_color():
    spheres = SpherePrimitiveBatch(
        numpy.array(((1, 0, 0), (0, 2, 0)), numpy.float32),
        numpy.array((1, 2), numpy.float32),
        numpy.array(((255, 0, 0, 255), (0, 255, 0, 128)), numpy.uint8))
    context = ExportGeometryContext(batch_vertex_limit=1024)
    meshes = list(spheres.export_geometry(context))

    assert len(meshes) == 1
    mesh = meshes[0]
    prototype_vertices = len(context.sphere_prototype(200)[0])
    assert len(mesh.vertices) == 2 * prototype_vertices
    assert numpy.all(mesh.vertex_colors[:prototype_vertices] == spheres.colors[0])
    assert numpy.all(mesh.vertex_colors[prototype_vertices:] == spheres.colors[1])
    assert spheres.centers.shape == (2, 3)


def test_sphere_geometry_filters_zero_radius_and_maps_picking_ids():
    spheres = SpherePrimitiveBatch(
        numpy.array(((0, 0, 0), (1, 2, 3)), numpy.float32),
        numpy.array((0, 2), numpy.float32),
        numpy.array(((1, 2, 3, 255), (4, 5, 6, 255)), numpy.uint8),
        picking_ids=numpy.array((3, 9), numpy.int32))

    assert spheres.primitive_count == 1
    assert spheres.picking_ids.tolist() == [9]
    bounds = spheres.bounds()
    assert numpy.allclose(bounds.xyz_min, (-1, 0, 1))
    assert numpy.allclose(bounds.xyz_max, (3, 4, 5))


def test_export_context_reuses_prototypes_and_bounds_batches():
    context = ExportGeometryContext()
    prototype = context.sphere_prototype(200)
    assert context.sphere_prototype(200) is prototype
    context.batch_vertex_limit = len(prototype[0])
    spheres = SpherePrimitiveBatch(
        numpy.zeros((3, 3), numpy.float32),
        numpy.ones((3,), numpy.float32),
        numpy.full((3, 4), 255, numpy.uint8))

    meshes = list(spheres.export_geometry(context))
    assert len(meshes) == 3
    assert all(len(mesh.vertices) <= context.batch_vertex_limit for mesh in meshes)


def test_cylinder_export_filters_degenerate_primitives_and_honors_caps():
    cylinders = CylinderPrimitiveBatch(
        numpy.array(((0, 0, 0), (1, 1, 1)), numpy.float32),
        numpy.array(((0, 0, 2), (1, 1, 1)), numpy.float32),
        numpy.array((0.5, 0.5), numpy.float32),
        numpy.array(((1, 2, 3, 255), (4, 5, 6, 255)), numpy.uint8),
        caps=numpy.array(((True, False), (True, True)), bool),
        picking_ids=numpy.array((7, 8), numpy.int32))

    assert cylinders.primitive_count == 1
    assert cylinders.picking_ids.tolist() == [7]
    endpoint = cylinders.endpoints[0]
    assert endpoint[3] == 1
    meshes = list(cylinders.export_geometry(ExportGeometryContext()))
    assert len(meshes) == 1
    assert len(meshes[0].triangles) == 3 * 15  # sides plus one cap


def test_default_triangle_drawing_uses_the_shared_export_contract():
    drawing = Drawing('triangles')
    vertices = numpy.array(((0, 0, 0), (1, 0, 0), (0, 1, 0)), numpy.float32)
    normals = numpy.array(((0, 0, 1),) * 3, numpy.float32)
    triangles = numpy.array(((0, 1, 2),), numpy.int32)
    drawing.set_geometry(vertices, normals, triangles)

    mesh = next(drawing.export_geometry(ExportGeometryContext()))
    assert mesh.vertices is vertices
    assert mesh.normals is normals
    assert mesh.triangles is drawing.masked_triangles
