from typing import Optional

import numpy as np

from chimerax.segmentations.segmentation import SegmentationStrategy

from chimerax.dicom.types import Axis


class PlanePuckSegmentation(SegmentationStrategy):
    def __init__(
        self,
        axis: Axis,
        plane: int,
        positions: list[(int, int)],
        value: int,
        min_threshold: float = -float("inf"),
        max_threshold: float = float("inf"),
    ):
        self.axis = axis
        self.plane = plane
        self.positions = positions
        self.value = value
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

    def execute(self, grid, reference_grid):
        for position in self.positions:
            center_x, center_y, radius = position
            self._set_data_in_puck(
                grid,
                reference_grid,
                self.axis,
                self.plane,
                round(center_x),
                round(center_y),
                radius,
                self.value,
                self.min_threshold,
                self.max_threshold,
            )

    def _set_data_in_puck(
        self,
        grid,
        reference_grid,
        axis: Axis,
        slice_number: int,
        left_offset: int,
        bottom_offset: int,
        radius: int,
        value: int,
        min_threshold: Optional[float] = None,
        max_threshold: Optional[float] = None,
    ) -> None:
        # TODO: if not segmentation, refuse
        # TODO: Preserve the happiest path. If the radius of the segmentation overlay is
        #  less than the radius of one voxel, there's no need to go through all the rigamarole.
        #  grid.data.segment_array[slice][left_offset][bottom_offset] = 1
        x_max, y_max, z_max = grid.size
        x_step, y_step, z_step = grid.step
        min_threshold = min_threshold if min_threshold is not None else float("-inf")
        max_threshold = max_threshold if max_threshold is not None else float("inf")
        if axis == Axis.AXIAL:
            slice = grid.array[slice_number]
            reference_slice = reference_grid.matrix()[slice_number]
            horizontal_max, vertical_max = x_max - 1, y_max - 1
            horizontal_step, vertical_step = x_step, y_step
        elif axis == Axis.CORONAL:
            slice = grid.array[:, slice_number, :]
            reference_slice = reference_grid.matrix()[:, slice_number, :]
            horizontal_max, vertical_max = x_max - 1, z_max - 1
            horizontal_step, vertical_step = x_step, z_step
        else:
            slice = grid.array[:, :, slice_number]
            reference_slice = reference_grid.matrix()[:, :, slice_number]
            horizontal_max, vertical_max = y_max - 1, z_max - 1
            horizontal_step, vertical_step = y_step, z_step

        scaled_radius = round(radius / horizontal_step)

        def apply_mask(x_start, x_end, y_start, y_end):
            """
            Apply the mask within bounds, checking thresholds.
            """
            # Apply thresholds and modify the slice
            for y_row in range(y_start, y_end + 1):
                mask = np.logical_and(
                    reference_slice[y_row, x_start:x_end + 1] >= min_threshold,
                    reference_slice[y_row, x_start:x_end + 1] <= max_threshold,
                )
                slice[y_row, x_start:x_end + 1][mask] = value

        x = 0
        y = round(radius)
        d = 1 - y
        while y > x:
            if d < 0:
                d += 2 * x + 3
            else:
                d += 2 * (x - y) + 5
                y -= 1
            x += 1
            scaled_horiz_x = round(x / horizontal_step)
            scaled_vert_x = round(x / vertical_step)
            scaled_horiz_y = round(y / horizontal_step)
            scaled_vert_y = round(y / vertical_step)

            x_start = round(max(left_offset - scaled_horiz_x, 0))
            x_end = round(min(left_offset + scaled_horiz_x, horizontal_max - 1))
            y_start = round(max(bottom_offset - scaled_vert_y, 0))
            y_end = round(min(bottom_offset + scaled_vert_y, vertical_max))
            apply_mask(x_start, x_end, y_start, y_end)

            x_start = round(max(left_offset - scaled_horiz_y, 0))
            x_end = round(min(left_offset + scaled_horiz_y, horizontal_max))
            y_start = round(max(bottom_offset - scaled_vert_x, 0))
            y_end = round(min(bottom_offset + scaled_vert_x, vertical_max))
            apply_mask(x_start, x_end, y_start, y_end)


class SphericalSegmentation(SegmentationStrategy):
    """Given a center, and radius, go to that point in some grid data and set all points
    insite the sphere to value. May also supply a minimum_threshold and a maximum_threshold;
    points will only be set if they are between the two values in the grid the segmentation
    is based on."""

    def __init__(
        self,
        center: tuple[int, int, int],
        radius: int,
        value: int,
        min_threshold: Optional[float] = None,
        max_threshold: Optional[float] = None,
    ):
        self.center = center
        self.radius = radius
        self.value = value
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

    def execute(self, grid, reference_grid):
        self._set_sphere_data(grid, reference_grid)

    def _set_sphere_data(self, grid, reference_grid) -> None:
        # Optimization: Mask only subregion containing sphere.
        ijk_min, ijk_max = self._sphere_grid_bounds(grid)
        from chimerax.map_data import GridSubregion, zone_mask

        subgrid = GridSubregion(grid, ijk_min, ijk_max)
        reference_subgrid = GridSubregion(reference_grid, ijk_min, ijk_max)

        if self.min_threshold and self.max_threshold:
            mask = zone_mask_clamped_by_referenced_grid(
                subgrid,
                reference_subgrid,
                [self.center],
                self.radius,
                self.min_threshold,
                self.max_threshold,
            )
        else:
            mask = zone_mask(subgrid, [self.center], self.radius)

        dmatrix = subgrid.full_matrix()

        from numpy import putmask

        putmask(dmatrix, mask, self.value)

    def _sphere_grid_bounds(self, grid):
        ijk_center = grid.xyz_to_ijk(self.center)
        spacings = grid.plane_spacings()
        ijk_size = [self.radius / s for s in spacings]
        from math import floor, ceil

        ijk_min = [max(int(floor(c - s)), 0) for c, s in zip(ijk_center, ijk_size)]
        ijk_max = [
            min(int(ceil(c + s)), m - 1)
            for c, s, m in zip(ijk_center, ijk_size, grid.size)
        ]
        return ijk_min, ijk_max


def zone_mask_clamped_by_referenced_grid(
    grid_data,
    referenced_grid,
    zone_points,
    zone_radius,
    min_value,
    max_value,
    invert_mask=False,
):
    """Like zone_mask from the map_data bundle, but with min/max value filtering"""
    from numpy import single as floatc, array, ndarray, zeros, int8, where

    if not isinstance(zone_points, ndarray):
        zone_points = array(zone_points, floatc)

    shape = tuple(reversed(grid_data.size))
    mask_3d = zeros(shape, int8)
    mask_1d = mask_3d.ravel()

    if invert_mask:
        mask_value = 0
        mask_1d[:] = 1
    else:
        mask_value = 1

    from chimerax.map_data import grid_indices
    from chimerax.geometry import find_closest_points

    size_limit = 2**22  # 4 Mvoxels
    if mask_3d.size > size_limit:
        # Calculate plane by plane to save memory with grid point array
        xsize, ysize, zsize = grid_data.size
        grid_points = grid_indices((xsize, ysize, 1), floatc)
        grid_data.ijk_to_xyz_transform.transform_points(grid_points, in_place=True)
        flat_reference = referenced_grid.full_matrix().ravel()
        zstep = [grid_data.ijk_to_xyz_transform.matrix[a][2] for a in range(3)]
        for z in range(zsize):
            i1, _, _ = find_closest_points(grid_points, zone_points, zone_radius)
            offset = xsize * ysize * z
            if min_value and max_value:
                mask = where(
                    (min_value <= flat_reference[i1 + offset])
                    & (flat_reference[i1 + offset] <= max_value),
                    1,
                    0,
                )
                mask_1d[i1 + offset] = mask
            else:
                mask_1d[i1 + offset] = mask_value
            grid_points[:, :] += zstep
    else:
        grid_points = grid_indices(grid_data.size, floatc)
        grid_data.ijk_to_xyz_transform.transform_points(grid_points, in_place=True)
        flat_reference = referenced_grid.full_matrix().ravel()
        i1, _, _ = find_closest_points(grid_points, zone_points, zone_radius)
        if min_value and max_value:
            mask = where(
                (min_value <= flat_reference[i1]) & (flat_reference[i1] <= max_value),
                1,
                0,
            )
            mask_1d[i1] = mask
        else:
            mask_1d[i1] = mask_value
    return mask_3d
