// vi: set expandtab shiftwidth=4 softtabstop=4:

/*
 * === UCSF ChimeraX Copyright ===
 * Copyright 2022-2025 Regents of the University of California. All rights
 * reserved. The ChimeraX application is provided pursuant to the ChimeraX
 * license agreement, which covers academic and commercial uses. For more
 * details, see <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
 *
 * This particular file is part of the ChimeraX library. You can also
 * redistribute and/or modify it under the terms of the GNU Lesser General
 * Public License version 2.1 as published by the Free Software Foundation.
 * For more details, see
 * <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
 *
 * THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
 * EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
 * LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
 * VERSION 2.1
 *
 * This notice must be embedded in or attached to all copies, including partial
 * copies, of the software or any revisions or derivations thereof.
 * === UCSF ChimeraX Copyright ===
 */

// ----------------------------------------------------------------------------
// Flying Edges algorithm for isosurface extraction
// A new implementation based closely on the VTK vtkFlyingEdges3D algorithm.
// See: Schroeder, W., Maynard, R., Geveci, B. (2015)
// "Flying Edges: A High-Performance Scalable Isocontouring Algorithm"
//
// Key insight: VTK uses a different edge numbering than standard MC.
// VTK groups edges by axis type for efficient Flying Edges traversal:
//   X-edges (0-3): edges parallel to X axis
//   Y-edges (4-7): edges parallel to Y axis
//   Z-edges (8-11): edges parallel to Z axis
//
// Standard MC edge numbering from contourdata.h:
//   Edge 0: (0,1) X-axis at origin
//   Edge 1: (1,2) Y-axis at x+1
//   Edge 2: (2,3) X-axis at y+1
//   Edge 3: (3,0) Y-axis at origin
//   Edge 4: (4,5) X-axis at z+1
//   Edge 5: (5,6) Y-axis at x+1,z+1
//   Edge 6: (6,7) X-axis at y+1,z+1
//   Edge 7: (7,4) Y-axis at z+1
//   Edge 8: (0,4) Z-axis at origin
//   Edge 9: (1,5) Z-axis at x+1
//   Edge 10: (2,6) Z-axis at x+1,y+1
//   Edge 11: (3,7) Z-axis at y+1
//
// VTK Flying Edges internal numbering:
//   FE 0: X-edge at origin = MC 0
//   FE 1: X-edge at y+1 = MC 2
//   FE 2: X-edge at z+1 = MC 4
//   FE 3: X-edge at y+1,z+1 = MC 6
//   FE 4: Y-edge at origin = MC 3
//   FE 5: Y-edge at x+1 = MC 1
//   FE 6: Y-edge at z+1 = MC 7
//   FE 7: Y-edge at x+1,z+1 = MC 5
//   FE 8: Z-edge at origin = MC 8
//   FE 9: Z-edge at x+1 = MC 9
//   FE 10: Z-edge at y+1 = MC 11
//   FE 11: Z-edge at x+1,y+1 = MC 10
// ----------------------------------------------------------------------------

#ifndef FLYING_EDGES_HEADER_INCLUDED
#define FLYING_EDGES_HEADER_INCLUDED

#include "contourdata.h" // For triangle case tables and cube_edges
#include "index_types.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

namespace FlyingEdges {

// A new, clean implementation of Flying Edges, based on VTK's battle-tested
// code.
template <class Data_Type> class FlyingEdgesSurface {
public:
  FlyingEdgesSurface(const Data_Type *grid, const AIndex size[3],
                     const GIndex stride[3], float threshold)

      : grid(grid), threshold(threshold), total_vertices(0), total_triangles(0),
        output_vertices(nullptr), output_triangles(nullptr),
        output_normals(nullptr) {
    for (int i = 0; i < 3; ++i) {
      this->size[i] = size[i];
      this->stride[i] = stride[i];
    }

    build_case_tables();

    // Execute passes 1-3 to count vertices and triangles
    // Pass 4 (geometry generation) is deferred to generate() method
    pass1_process_x_edges();
    pass2_process_yz_edges();
    pass3_prefix_sum();

    // Debug output
    // fprintf(stderr,
    //         "FlyingEdges: size=(%lld,%lld,%lld) threshold=%g vertices=%llu "
    //         "triangles=%llu\n",
    //         (long long)size[0], (long long)size[1], (long long)size[2],
    //         threshold, (unsigned long long)total_vertices,
    //         (unsigned long long)total_triangles);
  };

  ~FlyingEdgesSurface() {};

  VIndex vertex_count() const { return total_vertices; }
  TIndex triangle_count() const { return total_triangles; }

  // Generate geometry directly into caller-provided buffers (no copy!)
  void generate(float *vertex_xyz, VIndex *triangle_vertex_indices) {
    output_vertices = vertex_xyz;
    output_triangles = triangle_vertex_indices;
    output_normals = nullptr; // No inline normals
    if (total_vertices > 0 && total_triangles > 0) {
      pass4_generate_output();
    }
    output_vertices = nullptr;
    output_triangles = nullptr;
  }

  // Generate geometry AND normals inline (optimized - better cache locality)
  void generate_with_normals(float *vertex_xyz, VIndex *triangle_vertex_indices,
                             float *normals_xyz) {
    output_vertices = vertex_xyz;
    output_triangles = triangle_vertex_indices;
    output_normals = normals_xyz; // Enable inline normal computation
    if (total_vertices > 0 && total_triangles > 0) {
      pass4_generate_output();
    }
    output_vertices = nullptr;
    output_triangles = nullptr;
    output_normals = nullptr;
  }

  // Legacy interface - calls generate()
  void geometry(float *vertex_xyz, VIndex *triangle_vertex_indices) {
    generate(vertex_xyz, triangle_vertex_indices);
  }

  // Compute normals from gradient at each vertex position (fallback when not
  // inlined)
  void normals(float *normals_xyz, const float *vertex_xyz) {
    // #pragma omp parallel for schedule(static)
    for (VIndex v = 0; v < total_vertices; ++v) {
      float x = vertex_xyz[v * 3 + 0];
      float y = vertex_xyz[v * 3 + 1];
      float z = vertex_xyz[v * 3 + 2];
      float g[3];
      compute_gradient(x, y, z, g);
      float len = sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]);
      if (len > 0) {
        normals_xyz[v * 3 + 0] = -g[0] / len;
        normals_xyz[v * 3 + 1] = -g[1] / len;
        normals_xyz[v * 3 + 2] = -g[2] / len;
      } else {
        normals_xyz[v * 3 + 0] = 0;
        normals_xyz[v * 3 + 1] = 0;
        normals_xyz[v * 3 + 2] = 1;
      }
    }
  }

private:
  // Input data
  const Data_Type *grid;
  AIndex size[3];
  GIndex stride[3];
  float threshold;

  // Per-x-row metadata. Stores number of intersections on x,y,z axes,
  // number of triangles, and trim boundaries. In Pass 3, the intersection
  // counts are converted to offsets.
  struct EdgeMetaData {
    VIndex x_ints, y_ints, z_ints;
    TIndex num_tris;
    AIndex x_min, x_max;
  };
  std::vector<unsigned char> x_cases; // Stores classification for all x-edges
  std::vector<EdgeMetaData> edge_meta_data; // One per x-row in the volume

  // Output data
  VIndex total_vertices;
  TIndex total_triangles;

  // Output pointers (set during generate(), used by pass4)
  float *output_vertices;
  VIndex *output_triangles;
  float *output_normals; // Optional: if set, normals are computed inline during
                         // pass4

  // Flying Edges case table (FE edge numbering) and edge uses
  // EdgeCases[eCase][0] = number of triangles
  // EdgeCases[eCase][1..] = FE edge indices for triangles
  unsigned char EdgeCases[256][16];

  // EdgeUses[eCase][fe_edge] = 1 if FE edge is used for this case
  unsigned char EdgeUses[256][12];

  // Pre-computed edge grid offsets for fast interpolation in Pass 4
  // EdgeOffsets[feEdge][0/1] = grid offset for vertex 0/1 of edge
  GIndex edge_grid_offsets[12][2];

  // Maps MC edge numbers to FE edge numbers (from VTK's EdgeMap)
  static constexpr unsigned char MCtoFE[12] = {0, 5, 1, 4, 2,  7,
                                               3, 6, 8, 9, 11, 10};

  // Maps FE edge numbers to MC edge numbers (inverse of MCtoFE)
  static constexpr unsigned char FEtoMC[12] = {0, 2, 4, 6, 3,  1,
                                               7, 5, 8, 9, 11, 10};

  // Vertex offsets in (i,j,k) for each of 8 voxel corners
  // Using FE vertex numbering: vertices are ordered as we sweep through x-edges
  // FE vertex numbering from VTK:
  //   0: (0,0,0), 1: (1,0,0), 2: (0,1,0), 3: (1,1,0)
  //   4: (0,0,1), 5: (1,0,1), 6: (0,1,1), 7: (1,1,1)
  static constexpr unsigned char VertOffsets[8][3] = {
      {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0},
      {0, 0, 1}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}};

  // FE edge endpoint vertices (in FE vertex numbering)
  static constexpr unsigned char FEVertMap[12][2] = {
      {0, 1}, {2, 3}, {4, 5}, {6, 7}, // X-edges 0-3
      {0, 2}, {1, 3}, {4, 6}, {5, 7}, // Y-edges 4-7
      {0, 4}, {1, 5}, {2, 6}, {3, 7}  // Z-edges 8-11
  };

  // Boundary location flags (matching VTK's scheme)
  enum CellClass { Interior = 0, MinBoundary = 1, MaxBoundary = 2 };

  void build_case_tables() {
    // Build FE case tables from MC triangle_table
    // We need to convert FE edge-based cases to MC vertex-based cases
    // and map MC edge numbers to FE edge numbers

    std::memset(EdgeCases, 0, sizeof(EdgeCases));
    std::memset(EdgeUses, 0, sizeof(EdgeUses));

    // vertMap maps MC vertex index to which bit in eCase represents that vertex
    // MC vertices: 0(0,0,0), 1(1,0,0), 2(1,1,0), 3(0,1,0), 4(0,0,1), 5(1,0,1),
    // 6(1,1,1), 7(0,1,1) FE vertices: 0(0,0,0), 1(1,0,0), 2(0,1,0), 3(1,1,0),
    // 4(0,0,1), 5(1,0,1), 6(0,1,1), 7(1,1,1) eCase bits: 0=FEv0, 1=FEv1,
    // 2=FEv2, 3=FEv3, 4=FEv4, 5=FEv5, 6=FEv6, 7=FEv7 So vertMap[mc_vert] =
    // fe_vert where fe_vert is at same position as mc_vert
    static const int vertMap[8] = {0, 1, 3, 2, 4, 5, 7, 6};
    static const int CASE_MASK[8] = {1, 2, 4, 8, 16, 32, 64, 128};

    // Build edge case table
    // eCase ranges from 0 to 255, with each bit representing whether a FE
    // vertex is above threshold
    for (int eCase = 0; eCase < 256; ++eCase) {
      // Convert FE edge case to MC vertex case
      // For each MC vertex ii, check if bit vertMap[ii] is set in eCase
      int mcCase = 0;
      for (int ii = 0; ii < 8; ++ii) {
        if (eCase & (1 << vertMap[ii])) {
          mcCase |= CASE_MASK[ii];
        }
      }

      // Look up triangles from MC table
      const int *edges = triangle_table[mcCase];
      int numTris = 0;
      for (int t = 0; edges[t] != -1; t += 3) {
        numTris++;
      }

      EdgeCases[eCase][0] = numTris;

      // Store edge indices (converted to FE numbering)
      unsigned char *edgeCasePtr = &EdgeCases[eCase][1];
      for (int t = 0; edges[t] != -1; t += 3) {
        edgeCasePtr[0] = MCtoFE[edges[t]];
        edgeCasePtr[1] = MCtoFE[edges[t + 1]];
        edgeCasePtr[2] = MCtoFE[edges[t + 2]];
        edgeCasePtr += 3;
      }

      // Mark which FE edges are used
      for (int t = 0; t < numTris * 3; ++t) {
        EdgeUses[eCase][EdgeCases[eCase][1 + t]] = 1;
      }
    }
  };

  // Get the edge case for a voxel from 4 x-edge cases
  unsigned char getEdgeCase(unsigned char *ePtr[4]) {
    return (*ePtr[0]) | ((*ePtr[1]) << 2) | ((*ePtr[2]) << 4) |
           ((*ePtr[3]) << 6);
  }

  // Count boundary Y/Z intersections
  void countBoundaryYZInts(unsigned char loc, unsigned char *edgeUses,
                           EdgeMetaData *eMD[4]) {
    // loc encodes boundary position: bits 0-1 = x, bits 2-3 = y, bits 4-5 = z
    // Each pair: 0=interior, 1=min boundary, 2=max boundary
    switch (loc) {
    case 2:                          // +x boundary
      eMD[0]->y_ints += edgeUses[5]; // FE edge 5 = Y-edge at x+1
      eMD[0]->z_ints += edgeUses[9]; // FE edge 9 = Z-edge at x+1
      break;
    case 8:                           // +y boundary
      eMD[1]->z_ints += edgeUses[10]; // FE edge 10 = Z-edge at y+1
      break;
    case 10: // +x +y boundary
      eMD[0]->y_ints += edgeUses[5];
      eMD[0]->z_ints += edgeUses[9];
      eMD[1]->z_ints += edgeUses[10];
      eMD[1]->z_ints += edgeUses[11]; // FE edge 11 = Z-edge at x+1,y+1
      break;
    case 32:                         // +z boundary
      eMD[2]->y_ints += edgeUses[6]; // FE edge 6 = Y-edge at z+1
      break;
    case 34: // +x +z boundary
      eMD[0]->y_ints += edgeUses[5];
      eMD[0]->z_ints += edgeUses[9];
      eMD[2]->y_ints += edgeUses[6];
      eMD[2]->y_ints += edgeUses[7]; // FE edge 7 = Y-edge at x+1,z+1
      break;
    case 40: // +y +z boundary
      eMD[1]->z_ints += edgeUses[10];
      eMD[2]->y_ints += edgeUses[6];
      break;
    case 42: // +x +y +z boundary (corner)
      eMD[0]->y_ints += edgeUses[5];
      eMD[0]->z_ints += edgeUses[9];
      eMD[1]->z_ints += edgeUses[10];
      eMD[1]->z_ints += edgeUses[11];
      eMD[2]->y_ints += edgeUses[6];
      eMD[2]->y_ints += edgeUses[7];
      break;
    default:
      break;
    }
  }

  // Initialize voxel edge IDs at the start of a row
  unsigned char initVoxelIds(unsigned char *ePtr[4], EdgeMetaData *eMD[4],
                             VIndex *eIds) {
    unsigned char eCase = getEdgeCase(ePtr);
    // X-edges: one per row
    eIds[0] = eMD[0]->x_ints; // FE edge 0: x-edge at (j,k)
    eIds[1] = eMD[1]->x_ints; // FE edge 1: x-edge at (j+1,k)
    eIds[2] = eMD[2]->x_ints; // FE edge 2: x-edge at (j,k+1)
    eIds[3] = eMD[3]->x_ints; // FE edge 3: x-edge at (j+1,k+1)
    // Y-edges
    eIds[4] = eMD[0]->y_ints;               // FE edge 4: y-edge at origin
    eIds[5] = eIds[4] + EdgeUses[eCase][4]; // FE edge 5: y-edge at x+1
    eIds[6] = eMD[2]->y_ints;               // FE edge 6: y-edge at z+1
    eIds[7] = eIds[6] + EdgeUses[eCase][6]; // FE edge 7: y-edge at x+1,z+1
    // Z-edges
    eIds[8] = eMD[0]->z_ints;                  // FE edge 8: z-edge at origin
    eIds[9] = eIds[8] + EdgeUses[eCase][8];    // FE edge 9: z-edge at x+1
    eIds[10] = eMD[1]->z_ints;                 // FE edge 10: z-edge at y+1
    eIds[11] = eIds[10] + EdgeUses[eCase][10]; // FE edge 11: z-edge at x+1,y+1
    return eCase;
  }

  // Advance voxel edge IDs along the x-row
  void advanceVoxelIds(unsigned char eCase, VIndex *eIds) {
    // X-edges advance by their usage count
    eIds[0] += EdgeUses[eCase][0];
    eIds[1] += EdgeUses[eCase][1];
    eIds[2] += EdgeUses[eCase][2];
    eIds[3] += EdgeUses[eCase][3];
    // Y-edges: origin edges advance, +x edges are recalculated
    eIds[4] += EdgeUses[eCase][4];
    eIds[5] = eIds[4] + EdgeUses[eCase][5];
    eIds[6] += EdgeUses[eCase][6];
    eIds[7] = eIds[6] + EdgeUses[eCase][7];
    // Z-edges: same pattern
    eIds[8] += EdgeUses[eCase][8];
    eIds[9] = eIds[8] + EdgeUses[eCase][9];
    eIds[10] += EdgeUses[eCase][10];
    eIds[11] = eIds[10] + EdgeUses[eCase][11];
  }

  // PASS 1: Process all x-edges
  void pass1_process_x_edges() {
    const AIndex nx = size[0], ny = size[1], nz = size[2];
    const AIndex num_x_rows = ny * nz;
    const AIndex num_x_edges_per_row = nx - 1;

    x_cases.resize(num_x_rows * num_x_edges_per_row);
    edge_meta_data.resize(num_x_rows);

    const GIndex xstride = stride[0];
    const GIndex ystride = stride[1];
    const GIndex zstride = stride[2];
    const Data_Type thresh = static_cast<Data_Type>(threshold);

    // #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t k = 0; k < nz; ++k) {
      for (int64_t j = 0; j < ny; ++j) {
        const AIndex row_index = k * ny + j;
        EdgeMetaData &meta = edge_meta_data[row_index];
        meta.x_ints = meta.y_ints = meta.z_ints = 0;
        meta.num_tris = 0;
        meta.x_min = num_x_edges_per_row;
        meta.x_max = 0;

        unsigned char *row_cases = &x_cases[row_index * num_x_edges_per_row];

        // Pointer-based traversal along x-row - avoid repeated get_value()
        // calls
        const Data_Type *row_ptr = grid + j * ystride + k * zstride;
        Data_Type s0 = row_ptr[0];
        unsigned char above0 = (s0 >= thresh) ? 1 : 0;

        for (AIndex i = 0; i < num_x_edges_per_row; ++i) {
          Data_Type s1 = row_ptr[(i + 1) * xstride];
          unsigned char above1 = (s1 >= thresh) ? 1 : 0;

          // Edge case: bit 0 = left vertex above, bit 1 = right vertex above
          unsigned char edge_case = above0 | (above1 << 1);
          row_cases[i] = edge_case;

          if (edge_case == 1 || edge_case == 2) { // Edge is cut
            meta.x_ints++;
            if (i < meta.x_min)
              meta.x_min = i;
            if (i + 1 > meta.x_max)
              meta.x_max = i + 1;
          }

          // Carry forward for next iteration
          above0 = above1;
        }
      }
    }
  };

  // PASS 2: Count Y/Z edge intersections and triangles
  void pass2_process_yz_edges() {
    const AIndex nx = size[0], ny = size[1], nz = size[2];
    const AIndex num_x_edges_per_row = nx - 1;
    const AIndex num_x_rows = ny * nz;

    // Thread-local storage for counts - each thread gets its own copy
    // Will be reduced into edge_meta_data at the end
    // #pragma omp parallel
    {
      // Thread-local accumulators (zeroed)
      std::vector<VIndex> local_y_ints(num_x_rows, 0);
      std::vector<VIndex> local_z_ints(num_x_rows, 0);
      std::vector<TIndex> local_num_tris(num_x_rows, 0);

      // #pragma omp for collapse(2) schedule(static) nowait
      for (int64_t k = 0; k < (int64_t)(nz - 1); ++k) {
        for (int64_t j = 0; j < (int64_t)(ny - 1); ++j) {
          // Row indices for the 4 x-rows bounding this voxel row
          const AIndex idx0 = (k * ny) + j;
          const AIndex idx1 = (k * ny) + j + 1;
          const AIndex idx2 = ((k + 1) * ny) + j;
          const AIndex idx3 = ((k + 1) * ny) + j + 1;

          // Get metadata for trim boundaries (read-only here)
          EdgeMetaData *eMD[4];
          eMD[0] = &edge_meta_data[idx0];
          eMD[1] = &edge_meta_data[idx1];
          eMD[2] = &edge_meta_data[idx2];
          eMD[3] = &edge_meta_data[idx3];

          // Get x-edge case arrays for the 4 rows
          unsigned char *cases[4];
          cases[0] = &x_cases[idx0 * num_x_edges_per_row];
          cases[1] = &x_cases[idx1 * num_x_edges_per_row];
          cases[2] = &x_cases[idx2 * num_x_edges_per_row];
          cases[3] = &x_cases[idx3 * num_x_edges_per_row];

          // Compute trim boundaries from the 4 bounding x-rows
          AIndex xL = eMD[0]->x_min;
          AIndex xR = eMD[0]->x_max;
          if (eMD[1]->x_min < xL)
            xL = eMD[1]->x_min;
          if (eMD[2]->x_min < xL)
            xL = eMD[2]->x_min;
          if (eMD[3]->x_min < xL)
            xL = eMD[3]->x_min;
          if (eMD[1]->x_max > xR)
            xR = eMD[1]->x_max;
          if (eMD[2]->x_max > xR)
            xR = eMD[2]->x_max;
          if (eMD[3]->x_max > xR)
            xR = eMD[3]->x_max;

          // Skip if no intersections in any of the 4 rows
          if (xL >= xR) {
            unsigned char c0 = cases[0][0] & 1;
            unsigned char c1 = cases[1][0] & 1;
            unsigned char c2 = cases[2][0] & 1;
            unsigned char c3 = cases[3][0] & 1;
            if (c0 == c1 && c1 == c2 && c2 == c3) {
              continue;
            }
            xL = 0;
            xR = num_x_edges_per_row;
          }

          // Adjust trim boundaries
          if (xL > 0) {
            unsigned char c0 = (cases[0][xL - 1] >> 1) & 1;
            unsigned char c1 = (cases[1][xL - 1] >> 1) & 1;
            unsigned char c2 = (cases[2][xL - 1] >> 1) & 1;
            unsigned char c3 = (cases[3][xL - 1] >> 1) & 1;
            if (c0 != c1 || c0 != c2 || c0 != c3) {
              xL = 0;
            }
          }
          if (xR < num_x_edges_per_row) {
            unsigned char c0 = cases[0][xR] & 1;
            unsigned char c1 = cases[1][xR] & 1;
            unsigned char c2 = cases[2][xR] & 1;
            unsigned char c3 = cases[3][xR] & 1;
            if (c0 != c1 || c0 != c2 || c0 != c3) {
              xR = num_x_edges_per_row;
            }
          }

          // Store trim boundaries (only row 0 stores this, no conflict)
          eMD[0]->x_min = xL;
          eMD[0]->x_max = xR;

          // Process voxels in the row
          unsigned char *ePtr[4] = {cases[0] + xL, cases[1] + xL, cases[2] + xL,
                                    cases[3] + xL};
          const AIndex dim0Wall = nx - 2;
          unsigned char yLoc = (j >= (AIndex)(ny - 2) ? MaxBoundary : Interior);
          unsigned char zLoc = (k >= (AIndex)(nz - 2) ? MaxBoundary : Interior);
          unsigned char yzLoc = (yLoc << 2) | (zLoc << 4);

          for (AIndex i = xL; i < xR; ++i) {
            unsigned char eCase = getEdgeCase(ePtr);
            unsigned char numTris = EdgeCases[eCase][0];

            if (numTris > 0) {
              local_num_tris[idx0] += numTris;

              unsigned char *edgeUses = EdgeUses[eCase];
              // Count Y and Z axes edges (at voxel origin)
              local_y_ints[idx0] += edgeUses[4];
              local_z_ints[idx0] += edgeUses[8];

              // Count boundary edges into appropriate rows
              unsigned char loc =
                  yzLoc | (i >= dim0Wall ? MaxBoundary : Interior);
              if (loc != 0) {
                // Inline boundary counting to use local arrays
                if (loc & MaxBoundary) { // +x
                  local_y_ints[idx0] += edgeUses[5];
                  local_z_ints[idx0] += edgeUses[9];
                }
                if (loc & (MaxBoundary << 2)) { // +y
                  local_z_ints[idx1] += edgeUses[10];
                  if (loc & MaxBoundary) // +x +y
                    local_z_ints[idx1] += edgeUses[11];
                }
                if (loc & (MaxBoundary << 4)) { // +z
                  local_y_ints[idx2] += edgeUses[6];
                  if (loc & MaxBoundary) // +x +z
                    local_y_ints[idx2] += edgeUses[7];
                }
              }
            }

            ePtr[0]++;
            ePtr[1]++;
            ePtr[2]++;
            ePtr[3]++;
          }
        }
      }

      // Reduce thread-local counts into global edge_meta_data
      // #pragma omp critical
      {
        for (AIndex i = 0; i < num_x_rows; ++i) {
          edge_meta_data[i].y_ints += local_y_ints[i];
          edge_meta_data[i].z_ints += local_z_ints[i];
          edge_meta_data[i].num_tris += local_num_tris[i];
        }
      }
    } // end parallel
  };

  // PASS 3: Prefix sum to compute offsets
  // Uses VTK's per-row interleaved layout:
  // [row0 x-verts][row0 y-verts][row0 z-verts][row1 x-verts][row1 y-verts]...
  void pass3_prefix_sum() {
    VIndex numOutXPts = 0, numOutYPts = 0, numOutZPts = 0;
    TIndex numOutTris = 0;

    for (auto &meta : edge_meta_data) {
      VIndex numXPts = meta.x_ints;
      VIndex numYPts = meta.y_ints;
      VIndex numZPts = meta.z_ints;
      TIndex numTris = meta.num_tris;

      // Convert counts to offsets (VTK style)
      meta.x_ints = numOutXPts + numOutYPts +
                    numOutZPts; // Global start for this row's x-verts
      meta.y_ints = meta.x_ints + numXPts; // y-verts follow this row's x-verts
      meta.z_ints = meta.y_ints + numYPts; // z-verts follow this row's y-verts
      meta.num_tris = numOutTris;

      numOutXPts += numXPts;
      numOutYPts += numYPts;
      numOutZPts += numZPts;
      numOutTris += numTris;
    }

    total_vertices = numOutXPts + numOutYPts + numOutZPts;
    total_triangles = numOutTris;
    // Note: Buffer allocation is now done by caller before calling generate()
  };

  // Compute edge grid offsets for fast interpolation (called once before Pass
  // 4)
  void compute_edge_offsets() {
    for (int e = 0; e < 12; ++e) {
      unsigned char v0_idx = FEVertMap[e][0];
      unsigned char v1_idx = FEVertMap[e][1];
      edge_grid_offsets[e][0] = VertOffsets[v0_idx][0] * stride[0] +
                                VertOffsets[v0_idx][1] * stride[1] +
                                VertOffsets[v0_idx][2] * stride[2];
      edge_grid_offsets[e][1] = VertOffsets[v1_idx][0] * stride[0] +
                                VertOffsets[v1_idx][1] * stride[1] +
                                VertOffsets[v1_idx][2] * stride[2];
    }
  }

  // Interpolate a vertex on an edge - optimized version
  // Uses pre-computed offsets and pointer arithmetic
  inline void interpolateEdge(const Data_Type *voxel_ptr, AIndex i, AIndex j,
                              AIndex k, int feEdge, float *vert_ptr) {
    Data_Type s0 = voxel_ptr[edge_grid_offsets[feEdge][0]];
    Data_Type s1 = voxel_ptr[edge_grid_offsets[feEdge][1]];

    float t;
    float diff = (float)s1 - (float)s0;
    if (diff != 0.0f) {
      t = (threshold - (float)s0) / diff;
      // Clamp to [0, 1]
      t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
    } else {
      t = 0.5f;
    }

    // FEVertMap gives us which axis this edge is on:
    // Edges 0-3: X-axis (v0 and v1 differ in x)
    // Edges 4-7: Y-axis (v0 and v1 differ in y)
    // Edges 8-11: Z-axis (v0 and v1 differ in z)
    unsigned char v0_idx = FEVertMap[feEdge][0];
    float x0 = (float)(i + VertOffsets[v0_idx][0]);
    float y0 = (float)(j + VertOffsets[v0_idx][1]);
    float z0 = (float)(k + VertOffsets[v0_idx][2]);

    // Only one coordinate changes along the edge
    if (feEdge < 4) {
      // X-edge
      vert_ptr[0] = x0 + t;
      vert_ptr[1] = y0;
      vert_ptr[2] = z0;
    } else if (feEdge < 8) {
      // Y-edge
      vert_ptr[0] = x0;
      vert_ptr[1] = y0 + t;
      vert_ptr[2] = z0;
    } else {
      // Z-edge
      vert_ptr[0] = x0;
      vert_ptr[1] = y0;
      vert_ptr[2] = z0 + t;
    }
  }

  // Interpolate vertex AND compute normal inline (better cache locality)
  // The voxel data is already in cache from pass4 traversal
  inline void interpolateEdgeWithNormal(const Data_Type *voxel_ptr, AIndex i,
                                        AIndex j, AIndex k, int feEdge,
                                        float *vert_ptr, float *norm_ptr) {
    Data_Type s0 = voxel_ptr[edge_grid_offsets[feEdge][0]];
    Data_Type s1 = voxel_ptr[edge_grid_offsets[feEdge][1]];

    float t;
    float diff = (float)s1 - (float)s0;
    if (diff != 0.0f) {
      t = (threshold - (float)s0) / diff;
      t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
    } else {
      t = 0.5f;
    }

    // Compute vertex position
    unsigned char v0_idx = FEVertMap[feEdge][0];
    float x = (float)(i + VertOffsets[v0_idx][0]);
    float y = (float)(j + VertOffsets[v0_idx][1]);
    float z = (float)(k + VertOffsets[v0_idx][2]);

    if (feEdge < 4) {
      x += t;
    } else if (feEdge < 8) {
      y += t;
    } else {
      z += t;
    }

    vert_ptr[0] = x;
    vert_ptr[1] = y;
    vert_ptr[2] = z;

    // Compute gradient using same method as marching cubes (symmetric
    // differences)
    float g[3];
    compute_gradient(x, y, z, g);

    // Normalize (negative gradient points outward)
    float len = sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]);
    if (len > 0) {
      norm_ptr[0] = -g[0] / len;
      norm_ptr[1] = -g[1] / len;
      norm_ptr[2] = -g[2] / len;
    } else {
      norm_ptr[0] = 0;
      norm_ptr[1] = 0;
      norm_ptr[2] = 1;
    }
  }

  // Helper to emit vertex (and optionally normal) - dispatches based on
  // output_normals
  inline void emitVertex(const Data_Type *voxel_ptr, AIndex i, AIndex j,
                         AIndex k, int feEdge, VIndex vid) {
    if (output_normals) {
      interpolateEdgeWithNormal(voxel_ptr, i, j, k, feEdge,
                                &output_vertices[vid * 3],
                                &output_normals[vid * 3]);
    } else {
      interpolateEdge(voxel_ptr, i, j, k, feEdge, &output_vertices[vid * 3]);
    }
  }

  // PASS 4: Generate output geometry
  void pass4_generate_output() {
    if (total_vertices == 0 || total_triangles == 0) {
      return;
    }

    // Pre-compute edge grid offsets for fast interpolation
    compute_edge_offsets();

    const AIndex nx = size[0], ny = size[1], nz = size[2];
    const AIndex num_x_edges_per_row = nx - 1;
    const GIndex xstride = stride[0];
    const GIndex ystride = stride[1];
    const GIndex zstride = stride[2];

    // #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t k = 0; k < (int64_t)(nz - 1); ++k) {
      for (int64_t j = 0; j < (int64_t)(ny - 1); ++j) {
        // Get metadata for the 4 x-rows
        EdgeMetaData *eMD[4];
        eMD[0] = &edge_meta_data[(k * ny) + j];
        eMD[1] = &edge_meta_data[(k * ny) + j + 1];
        eMD[2] = &edge_meta_data[((k + 1) * ny) + j];
        eMD[3] = &edge_meta_data[((k + 1) * ny) + j + 1];

        // Check if there are any triangles in this row
        // Compare with next row's triangle offset
        TIndex nextRowTris;
        if (j + 1 < ny - 1) {
          nextRowTris = edge_meta_data[(k * ny) + j + 1].num_tris;
        } else if (k + 1 < nz - 1) {
          nextRowTris = edge_meta_data[((k + 1) * ny)].num_tris;
        } else {
          nextRowTris = total_triangles;
        }
        if (eMD[0]->num_tris >= nextRowTris) {
          continue; // No triangles in this row
        }

        // Get trim boundaries - DISABLED for now to fix vertex ID bug
        // The issue: when xL > 0, vertices before xL are counted but not
        // interpolated, causing triangles to reference uninitialized vertex
        // slots (at origin)
        AIndex xL = 0;                   // was: eMD[0]->x_min;
        AIndex xR = num_x_edges_per_row; // was: eMD[0]->x_max;
        // Skip completely empty rows
        if (eMD[0]->x_min >= eMD[0]->x_max)
          continue;

        // Get x-edge case arrays
        unsigned char *cases[4];
        cases[0] = &x_cases[((k * ny) + j) * num_x_edges_per_row];
        cases[1] = &x_cases[((k * ny) + j + 1) * num_x_edges_per_row];
        cases[2] = &x_cases[(((k + 1) * ny) + j) * num_x_edges_per_row];
        cases[3] = &x_cases[(((k + 1) * ny) + j + 1) * num_x_edges_per_row];

        unsigned char *ePtr[4] = {cases[0] + xL, cases[1] + xL, cases[2] + xL,
                                  cases[3] + xL};

        // Initialize edge IDs
        VIndex eIds[12];
        unsigned char eCase = initVoxelIds(ePtr, eMD, eIds);

        // Determine if this row is on a boundary
        const bool onYBoundary = (j >= (AIndex)(ny - 2));
        const bool onZBoundary = (k >= (AIndex)(nz - 2));
        const AIndex dim0Wall = nx - 2;

        VIndex *tri_ptr = output_triangles + eMD[0]->num_tris * 3;

        // Pre-compute base grid pointer for this voxel row
        const Data_Type *row_base = grid + j * ystride + k * zstride;

        // Fast path: interior rows (no boundary checks needed)
        if (!onYBoundary && !onZBoundary && xL > 0 && xR <= dim0Wall) {
          for (AIndex i = xL; i < xR; ++i) {
            unsigned char numTris = EdgeCases[eCase][0];
            if (numTris > 0) {
              const Data_Type *voxel_ptr = row_base + i * xstride;
              const unsigned char *edges = &EdgeCases[eCase][1];
              for (int t = 0; t < numTris; ++t) {
                tri_ptr[0] = eIds[edges[0]];
                tri_ptr[1] = eIds[edges[1]];
                tri_ptr[2] = eIds[edges[2]];
                tri_ptr += 3;
                edges += 3;
              }
              unsigned char *edgeUses = EdgeUses[eCase];
              if (edgeUses[0])
                emitVertex(voxel_ptr, i, j, k, 0, eIds[0]);
              if (edgeUses[4])
                emitVertex(voxel_ptr, i, j, k, 4, eIds[4]);
              if (edgeUses[8])
                emitVertex(voxel_ptr, i, j, k, 8, eIds[8]);
              // Advance edge IDs (only when triangles were generated, like VTK)
              advanceVoxelIds(eCase, eIds);
            }
            // Advance to next voxel
            if (i < xR - 1) {
              ePtr[0]++;
              ePtr[1]++;
              ePtr[2]++;
              ePtr[3]++;
              eCase = getEdgeCase(ePtr);
            }
          }
        } else {
          // Slow path: boundary rows need extra edge generation
          unsigned char yzLoc = (onYBoundary ? (MaxBoundary << 2) : 0) |
                                (onZBoundary ? (MaxBoundary << 4) : 0);

          for (AIndex i = xL; i < xR; ++i) {
            unsigned char numTris = EdgeCases[eCase][0];
            if (numTris > 0) {
              const Data_Type *voxel_ptr = row_base + i * xstride;
              const unsigned char *edges = &EdgeCases[eCase][1];
              for (int t = 0; t < numTris; ++t) {
                tri_ptr[0] = eIds[edges[0]];
                tri_ptr[1] = eIds[edges[1]];
                tri_ptr[2] = eIds[edges[2]];
                tri_ptr += 3;
                edges += 3;
              }
              unsigned char *edgeUses = EdgeUses[eCase];
              if (edgeUses[0])
                emitVertex(voxel_ptr, i, j, k, 0, eIds[0]);
              if (edgeUses[4])
                emitVertex(voxel_ptr, i, j, k, 4, eIds[4]);
              if (edgeUses[8])
                emitVertex(voxel_ptr, i, j, k, 8, eIds[8]);

              // Boundary edges
              unsigned char loc = yzLoc | (i >= dim0Wall ? MaxBoundary : 0);
              if (loc) {
                if ((loc & 0x3) == MaxBoundary) {
                  if (edgeUses[5])
                    emitVertex(voxel_ptr, i, j, k, 5, eIds[5]);
                  if (edgeUses[9])
                    emitVertex(voxel_ptr, i, j, k, 9, eIds[9]);
                }
                if (loc & (MaxBoundary << 2)) { // +y boundary
                  if (edgeUses[1])
                    emitVertex(voxel_ptr, i, j, k, 1, eIds[1]);
                  if (edgeUses[10])
                    emitVertex(voxel_ptr, i, j, k, 10, eIds[10]);
                  if ((loc & 0x3) == MaxBoundary && edgeUses[11])
                    emitVertex(voxel_ptr, i, j, k, 11, eIds[11]);
                }
                if (loc & (MaxBoundary << 4)) { // +z boundary
                  if (edgeUses[2])
                    emitVertex(voxel_ptr, i, j, k, 2, eIds[2]);
                  if (edgeUses[6])
                    emitVertex(voxel_ptr, i, j, k, 6, eIds[6]);
                  if ((loc & 0x3) == MaxBoundary) { // +x +z
                    // Edge 7 (y-edge at x+1,z+1) only on +x+z
                    if (edgeUses[7])
                      emitVertex(voxel_ptr, i, j, k, 7, eIds[7]);
                  }
                }
                // Edge 3 (x-edge at y+1,z+1) only on +y+z corner
                if ((loc & (MaxBoundary << 2)) && (loc & (MaxBoundary << 4))) {
                  if (edgeUses[3])
                    emitVertex(voxel_ptr, i, j, k, 3, eIds[3]);
                }
              }
              // Advance edge IDs (only when triangles were generated, like VTK)
              advanceVoxelIds(eCase, eIds);
            }
            // Advance to next voxel
            if (i < xR - 1) {
              ePtr[0]++;
              ePtr[1]++;
              ePtr[2]++;
              ePtr[3]++;
              eCase = getEdgeCase(ePtr);
            }
          }
        }
      }
    }
  };

  // Helper to get grid value
  inline Data_Type get_value(AIndex x, AIndex y, AIndex z) const {
    return grid[x * stride[0] + y * stride[1] + z * stride[2]];
  }

  // Compute gradient at a point (for normals)
  // Matches marching_cubes.h approach: uses symmetric/central differences
  void compute_gradient(float x, float y, float z, float *gradient) const {
    float pos[3] = {x, y, z};

    // Check for exact boundary positions first
    for (int a = 0; a < 3; ++a)
      gradient[a] = (pos[a] == 0 ? 1 : (pos[a] == size[a] - 1 ? -1 : 0));

    if (gradient[0] == 0 && gradient[1] == 0 && gradient[2] == 0) {
      // Interior point - use symmetric differences with interpolation
      AIndex i[3] = {(AIndex)pos[0], (AIndex)pos[1], (AIndex)pos[2]};
      const Data_Type *ga = grid + stride[0] * (GIndex)i[0] +
                            stride[1] * (GIndex)i[1] + stride[2] * (GIndex)i[2];
      const Data_Type *gb = ga;
      AIndex off[3] = {0, 0, 0};
      float fb = 0;

      // Find which axis has fractional component
      for (int a = 0; a < 3; ++a) {
        if ((fb = pos[a] - i[a]) > 0) {
          off[a] = 1;
          gb = ga + stride[a];
          break;
        }
      }
      float fa = 1 - fb;

      // Compute gradient using symmetric differences at each grid point
      for (int a = 0; a < 3; ++a) {
        GIndex s = stride[a];
        AIndex ia = i[a], ib = ia + off[a];
        // At boundary: forward/backward diff * 2, interior: central diff
        gradient[a] = fa * (ia == 0 ? 2 * ((float)ga[s] - ga[0])
                                    : (float)ga[s] - *(ga - s)) +
                      fb * (ib == 0             ? 2 * ((float)gb[s] - gb[0])
                            : ib == size[a] - 1 ? 2 * ((float)gb[0] - *(gb - s))
                                                : (float)gb[s] - *(gb - s));
      }
    }
  }
};

// Static member definitions
template <class Data_Type>
constexpr unsigned char FlyingEdgesSurface<Data_Type>::MCtoFE[12];

template <class Data_Type>
constexpr unsigned char FlyingEdgesSurface<Data_Type>::FEtoMC[12];

template <class Data_Type>
constexpr unsigned char FlyingEdgesSurface<Data_Type>::VertOffsets[8][3];

template <class Data_Type>
constexpr unsigned char FlyingEdgesSurface<Data_Type>::FEVertMap[12][2];

// ----------------------------------------------------------------------------
// Cap face generator - adds triangles to close the isosurface at volume
// boundaries This is a separate pass that runs after the main Flying Edges
// algorithm.
//
// Cap faces are needed when the isosurface extends to the boundary of the
// volume. Without caps, the surface would have holes at the boundaries.
//
// Algorithm:
// 1. Count cap vertices (boundary grid corners above threshold)
// 2. Count cap triangles (using marching squares on each boundary face)
// 3. Generate cap geometry
//
template <class Data_Type> class CapFaceGenerator {
public:
  CapFaceGenerator(const Data_Type *grid, const AIndex size[3],
                   const GIndex stride[3], float threshold)
      : grid(grid), threshold(threshold) {
    for (int i = 0; i < 3; ++i) {
      this->size[i] = size[i];
      this->stride[i] = stride[i];
    }
  }

  // Count cap vertices and triangles for all 6 boundary faces
  void count_caps(VIndex &cap_vertex_count, TIndex &cap_triangle_count) {
    cap_vertex_count = 0;
    cap_triangle_count = 0;

    // Process each of the 6 boundary faces
    // Face 0: k=0 (min Z), Face 1: j=0 (min Y), Face 2: i=size[0]-1 (max X)
    // Face 3: j=size[1]-1 (max Y), Face 4: i=0 (min X), Face 5: k=size[2]-1
    // (max Z)
    for (int face = 0; face < 6; ++face) {
      count_face_caps(face, cap_vertex_count, cap_triangle_count);
    }
  }

  // Generate cap geometry into provided buffers
  // cap_vertices starts after the main surface vertices
  // cap_triangles starts after the main surface triangles
  // cap_normals is optional - if non-null, generates face normals for cap
  // vertices
  void generate_caps(float *cap_vertices, VIndex *cap_triangles,
                     VIndex base_vertex_id, float *cap_normals = nullptr) {
    VIndex vertex_offset = 0;
    TIndex triangle_offset = 0;

    for (int face = 0; face < 6; ++face) {
      generate_face_caps(face, cap_vertices, cap_triangles, base_vertex_id,
                         vertex_offset, triangle_offset, cap_normals);
    }
  }

private:
  const Data_Type *grid;
  AIndex size[3];
  GIndex stride[3];
  float threshold;

  // Marching squares edge table for 2D cap face triangulation
  // Each of 16 cases lists edges that form the boundary, then corner vertices
  // to add to complete the cap triangles.
  // Format: number of triangles, then triangle vertex indices (edge 0-3 or
  // corner 4-7) Edge numbering on a face: 0=bottom, 1=right, 2=top, 3=left
  // Corner numbering: 4=bottom-left, 5=bottom-right, 6=top-right, 7=top-left

  // Get grid value at (i, j, k)
  inline Data_Type get_value(AIndex i, AIndex j, AIndex k) const {
    return grid[i * stride[0] + j * stride[1] + k * stride[2]];
  }

  // Get value on a 2D boundary face
  // face: 0=k_min, 1=j_min, 2=i_max, 3=j_max, 4=i_min, 5=k_max
  // u, v are the two varying coordinates on that face
  inline Data_Type get_face_value(int face, AIndex u, AIndex v) const {
    switch (face) {
    case 0:
      return get_value(u, v, 0); // k=0
    case 1:
      return get_value(u, 0, v); // j=0
    case 2:
      return get_value(size[0] - 1, u, v); // i=max
    case 3:
      return get_value(u, size[1] - 1, v); // j=max
    case 4:
      return get_value(0, u, v); // i=0
    case 5:
      return get_value(u, v, size[2] - 1); // k=max
    }
    return 0;
  }

  // Get 3D position from face coordinates
  inline void face_to_3d(int face, float u, float v, float *xyz) const {
    switch (face) {
    case 0:
      xyz[0] = u;
      xyz[1] = v;
      xyz[2] = 0;
      break;
    case 1:
      xyz[0] = u;
      xyz[1] = 0;
      xyz[2] = v;
      break;
    case 2:
      xyz[0] = size[0] - 1;
      xyz[1] = u;
      xyz[2] = v;
      break;
    case 3:
      xyz[0] = u;
      xyz[1] = size[1] - 1;
      xyz[2] = v;
      break;
    case 4:
      xyz[0] = 0;
      xyz[1] = u;
      xyz[2] = v;
      break;
    case 5:
      xyz[0] = u;
      xyz[1] = v;
      xyz[2] = size[2] - 1;
      break;
    }
  }

  // Get face dimensions (u_size, v_size)
  inline void get_face_dims(int face, AIndex &u_size, AIndex &v_size) const {
    switch (face) {
    case 0:
    case 5:
      u_size = size[0];
      v_size = size[1];
      break; // XY faces
    case 1:
    case 3:
      u_size = size[0];
      v_size = size[2];
      break; // XZ faces
    case 2:
    case 4:
      u_size = size[1];
      v_size = size[2];
      break; // YZ faces
    }
  }

  // Count caps for a single boundary face
  void count_face_caps(int face, VIndex &cap_verts, TIndex &cap_tris) {
    AIndex u_size, v_size;
    get_face_dims(face, u_size, v_size);

    if (u_size < 2 || v_size < 2)
      return;

    // For each cell on the face, check if it needs cap triangles
    for (AIndex v = 0; v < v_size - 1; ++v) {
      for (AIndex u = 0; u < u_size - 1; ++u) {
        // Get the 4 corner values
        bool c00 = get_face_value(face, u, v) >= threshold;
        bool c10 = get_face_value(face, u + 1, v) >= threshold;
        bool c11 = get_face_value(face, u + 1, v + 1) >= threshold;
        bool c01 = get_face_value(face, u, v + 1) >= threshold;

        int cell_case =
            (c00 ? 1 : 0) | (c10 ? 2 : 0) | (c11 ? 4 : 0) | (c01 ? 8 : 0);

        // Skip fully outside cells (no surface intersection)
        if (cell_case == 0)
          continue;

        // Count edge intersections (these become cap vertices)
        // Edge 0: bottom (c00-c10), Edge 1: right (c10-c11)
        // Edge 2: top (c11-c01), Edge 3: left (c01-c00)
        if (c00 != c10)
          cap_verts++; // Edge 0
        if (c10 != c11)
          cap_verts++; // Edge 1
        if (c11 != c01)
          cap_verts++; // Edge 2
        if (c01 != c00)
          cap_verts++; // Edge 3

        // Count corner vertices needed (corners that are inside)
        if (c00)
          cap_verts++;
        if (c10)
          cap_verts++;
        if (c11)
          cap_verts++;
        if (c01)
          cap_verts++;

        // Count triangles using marching squares table
        cap_tris += count_cell_triangles(cell_case);
      }
    }
  }

  // Count triangles for a marching squares case
  int count_cell_triangles(int cell_case) const {
    // Marching squares: cases with 1, 2, 3, or 4 corners inside
    // Each case has a specific number of triangles
    static const int tri_count[16] = {
        0, // 0: none inside
        1, // 1: c00 inside
        1, // 2: c10 inside
        2, // 3: c00, c10 inside
        1, // 4: c11 inside
        2, // 5: c00, c11 inside (ambiguous - use simple)
        2, // 6: c10, c11 inside
        3, // 7: c00, c10, c11 inside
        1, // 8: c01 inside
        2, // 9: c00, c01 inside
        2, // 10: c10, c01 inside (ambiguous - use simple)
        3, // 11: c00, c10, c01 inside
        2, // 12: c11, c01 inside
        3, // 13: c00, c11, c01 inside
        3, // 14: c10, c11, c01 inside
        2  // 15: all inside - need 2 triangles to fill the square
    };
    return tri_count[cell_case];
  }

  // Get the outward-facing normal for a boundary face
  inline void get_face_normal(int face, float *n) const {
    // Face 0: k=0 (min Z) -> normal (0,0,-1)
    // Face 1: j=0 (min Y) -> normal (0,-1,0)
    // Face 2: i=max (max X) -> normal (1,0,0)
    // Face 3: j=max (max Y) -> normal (0,1,0)
    // Face 4: i=0 (min X) -> normal (-1,0,0)
    // Face 5: k=max (max Z) -> normal (0,0,1)
    n[0] = n[1] = n[2] = 0;
    switch (face) {
    case 0:
      n[2] = -1;
      break; // min Z
    case 1:
      n[1] = -1;
      break; // min Y
    case 2:
      n[0] = 1;
      break; // max X
    case 3:
      n[1] = 1;
      break; // max Y
    case 4:
      n[0] = -1;
      break; // min X
    case 5:
      n[2] = 1;
      break; // max Z
    }
  }

  // Generate caps for a single boundary face
  void generate_face_caps(int face, float *vertices, VIndex *triangles,
                          VIndex base_vid, VIndex &v_off, TIndex &t_off,
                          float *normals) {
    AIndex u_size, v_size;
    get_face_dims(face, u_size, v_size);

    if (u_size < 2 || v_size < 2)
      return;

    // Get the face normal for cap vertices
    float face_normal[3];
    get_face_normal(face, face_normal);

    // Determine face normal direction for correct triangle winding
    // Based on how (u,v) maps to 3D for each face:
    // Face 0: (u,v)→(x,y,0), CCW→+Z, want -Z → flip
    // Face 1: (u,v)→(x,0,z), CCW→-Y, want -Y → no flip
    // Face 2: (u,v)→(max,y,z), CCW→+X, want +X → no flip
    // Face 3: (u,v)→(x,max,z), CCW→-Y, want +Y → flip
    // Face 4: (u,v)→(0,y,z), CCW→+X, want -X → flip
    // Face 5: (u,v)→(x,y,max), CCW→+Z, want +Z → no flip
    bool flip_winding = (face == 0 || face == 3 || face == 4);

    for (AIndex v = 0; v < v_size - 1; ++v) {
      for (AIndex u = 0; u < u_size - 1; ++u) {
        // Get the 4 corner values
        Data_Type v00 = get_face_value(face, u, v);
        Data_Type v10 = get_face_value(face, u + 1, v);
        Data_Type v11 = get_face_value(face, u + 1, v + 1);
        Data_Type v01 = get_face_value(face, u, v + 1);

        bool c00 = v00 >= threshold;
        bool c10 = v10 >= threshold;
        bool c11 = v11 >= threshold;
        bool c01 = v01 >= threshold;

        int cell_case =
            (c00 ? 1 : 0) | (c10 ? 2 : 0) | (c11 ? 4 : 0) | (c01 ? 8 : 0);

        // Skip fully outside cells (no cap needed)
        if (cell_case == 0)
          continue;

        // Create vertices for this cell
        // Local vertex indices: 0-3 = edge intersections, 4-7 = corners
        VIndex local_vids[8];
        int num_local_verts = 0;

        // Helper to emit a vertex (and optional normal)
        auto emit_vertex = [&](int local_idx, float fu, float fv) {
          float xyz[3];
          face_to_3d(face, fu, fv, xyz);
          VIndex idx = v_off + num_local_verts;
          vertices[idx * 3 + 0] = xyz[0];
          vertices[idx * 3 + 1] = xyz[1];
          vertices[idx * 3 + 2] = xyz[2];
          if (normals) {
            normals[idx * 3 + 0] = face_normal[0];
            normals[idx * 3 + 1] = face_normal[1];
            normals[idx * 3 + 2] = face_normal[2];
          }
          local_vids[local_idx] = base_vid + idx;
          num_local_verts++;
        };

        // Edge vertices (interpolated)
        if (c00 != c10) { // Edge 0 (bottom)
          float t = (threshold - (float)v00) / ((float)v10 - (float)v00);
          emit_vertex(0, u + t, (float)v);
        }
        if (c10 != c11) { // Edge 1 (right)
          float t = (threshold - (float)v10) / ((float)v11 - (float)v10);
          emit_vertex(1, (float)(u + 1), v + t);
        }
        if (c11 != c01) { // Edge 2 (top)
          float t = (threshold - (float)v01) / ((float)v11 - (float)v01);
          emit_vertex(2, u + t, (float)(v + 1));
        }
        if (c01 != c00) { // Edge 3 (left)
          float t = (threshold - (float)v00) / ((float)v01 - (float)v00);
          emit_vertex(3, (float)u, v + t);
        }

        // Corner vertices (at grid positions)
        if (c00)
          emit_vertex(4, (float)u, (float)v);
        if (c10)
          emit_vertex(5, (float)(u + 1), (float)v);
        if (c11)
          emit_vertex(6, (float)(u + 1), (float)(v + 1));
        if (c01)
          emit_vertex(7, (float)u, (float)(v + 1));

        v_off += num_local_verts;

        // Generate triangles for this cell case
        generate_cell_triangles(cell_case, local_vids, c00, c10, c11, c01,
                                triangles, t_off, flip_winding);
      }
    }
  }

  // Generate triangles for a marching squares cell
  void generate_cell_triangles(int cell_case, VIndex *vids, bool c00, bool c10,
                               bool c11, bool c01, VIndex *triangles,
                               TIndex &t_off, bool flip) {
    // Marching squares triangulation
    // vids[0-3] = edge vertices (if they exist)
    // vids[4-7] = corner vertices (if c00, c10, c11, c01 respectively)

    auto emit_tri = [&](int a, int b, int c) {
      if (flip) {
        triangles[t_off * 3 + 0] = vids[a];
        triangles[t_off * 3 + 1] = vids[c];
        triangles[t_off * 3 + 2] = vids[b];
      } else {
        triangles[t_off * 3 + 0] = vids[a];
        triangles[t_off * 3 + 1] = vids[b];
        triangles[t_off * 3 + 2] = vids[c];
      }
      t_off++;
    };

    switch (cell_case) {
    case 1: // c00 only
      emit_tri(3, 4, 0);
      break;
    case 2: // c10 only
      emit_tri(0, 5, 1);
      break;
    case 3: // c00, c10
      emit_tri(3, 4, 5);
      emit_tri(3, 5, 1);
      break;
    case 4: // c11 only
      emit_tri(1, 6, 2);
      break;
    case 5: // c00, c11 (diagonal)
      emit_tri(3, 4, 0);
      emit_tri(1, 6, 2);
      break;
    case 6: // c10, c11
      emit_tri(0, 5, 6);
      emit_tri(0, 6, 2);
      break;
    case 7: // c00, c10, c11
      emit_tri(3, 4, 5);
      emit_tri(3, 5, 6);
      emit_tri(3, 6, 2);
      break;
    case 8: // c01 only
      emit_tri(2, 7, 3);
      break;
    case 9: // c00, c01
      // Fixed winding: fan from corner 4
      emit_tri(4, 0, 2);
      emit_tri(4, 2, 7);
      break;
    case 10: // c10, c01 (diagonal)
      emit_tri(0, 5, 1);
      emit_tri(2, 7, 3);
      break;
    case 11: // c00, c10, c01
      emit_tri(1, 4, 5);
      emit_tri(1, 7, 4);
      emit_tri(1, 2, 7);
      break;
    case 12: // c11, c01
      emit_tri(1, 6, 7);
      emit_tri(1, 7, 3);
      break;
    case 13: // c00, c11, c01
      // Fixed winding: fan from corner 4
      emit_tri(4, 0, 1);
      emit_tri(4, 1, 6);
      emit_tri(4, 6, 7);
      break;
    case 14: // c10, c11, c01
      emit_tri(0, 5, 6);
      emit_tri(0, 6, 7);
      emit_tri(0, 7, 3);
      break;
    case 15: // all corners inside - fill the whole square
      // Vertices: 4=c00(0,0), 5=c10(1,0), 6=c11(1,1), 7=c01(0,1)
      // Two triangles to fill quad, CCW winding when viewed from +Z
      emit_tri(4, 5, 7);
      emit_tri(5, 6, 7);
      break;
      // case 0: no triangles (all outside)
    }
  }
};

// Factory wrapper class
template <class Data_Type>
class FlyingEdgesSurfaceWrapper : public Contour_Calculation::Contour_Surface {
public:
  FlyingEdgesSurfaceWrapper(const Data_Type *grid, const AIndex size[3],
                            const GIndex stride[3], float threshold,
                            bool cap_faces)
      : impl(grid, size, stride, threshold),
        cap_gen(grid, size, stride, threshold), do_cap_faces(cap_faces),
        last_vertices(nullptr), cached_cap_verts(0), cached_cap_tris(0) {
    // Count cap geometry if needed
    if (do_cap_faces) {
      cap_gen.count_caps(cached_cap_verts, cached_cap_tris);
    }
  }

  virtual ~FlyingEdgesSurfaceWrapper() {}

  virtual VIndex vertex_count() {
    return impl.vertex_count() + (do_cap_faces ? cached_cap_verts : 0);
  }
  virtual TIndex triangle_count() {
    return impl.triangle_count() + (do_cap_faces ? cached_cap_tris : 0);
  }
  virtual void geometry(float *vertex_xyz, VIndex *triangle_vertex_indices) {
    // Generate main surface
    impl.generate(vertex_xyz, triangle_vertex_indices);
    last_vertices = vertex_xyz;

    // Generate cap faces if enabled
    if (do_cap_faces && cached_cap_verts > 0) {
      VIndex base_verts = impl.vertex_count();
      TIndex base_tris = impl.triangle_count();
      cap_gen.generate_caps(vertex_xyz + base_verts * 3,
                            triangle_vertex_indices + base_tris * 3, base_verts,
                            nullptr);
    }
  }

  virtual void normals(float *normals) {
    // Compute normals for main surface
    impl.normals(normals, last_vertices);

    // For cap faces, we need to regenerate with normals since we didn't
    // store them during geometry(). This is slightly wasteful but correct.
    if (do_cap_faces && cached_cap_verts > 0) {
      VIndex base_verts = impl.vertex_count();
      TIndex base_tris = impl.triangle_count();
      // Create temporary buffers for re-generation
      std::vector<float> temp_verts(cached_cap_verts * 3);
      std::vector<VIndex> temp_tris(cached_cap_tris * 3);
      cap_gen.generate_caps(temp_verts.data(), temp_tris.data(), base_verts,
                            normals + base_verts * 3);
    }
  }

  virtual void geometry_with_normals(float *vertex_xyz,
                                     VIndex *triangle_vertex_indices,
                                     float *normals_xyz) {
    // Generate main surface with normals
    impl.generate_with_normals(vertex_xyz, triangle_vertex_indices,
                               normals_xyz);
    last_vertices = vertex_xyz;

    // Generate cap faces with normals
    if (do_cap_faces && cached_cap_verts > 0) {
      VIndex base_verts = impl.vertex_count();
      TIndex base_tris = impl.triangle_count();
      cap_gen.generate_caps(vertex_xyz + base_verts * 3,
                            triangle_vertex_indices + base_tris * 3, base_verts,
                            normals_xyz + base_verts * 3);
    }
  }

private:
  FlyingEdgesSurface<Data_Type> impl;
  CapFaceGenerator<Data_Type> cap_gen;
  bool do_cap_faces;
  float *last_vertices;
  VIndex cached_cap_verts;
  TIndex cached_cap_tris;
};

} // namespace FlyingEdges

#endif // FLYING_EDGES_HEADER_INCLUDED
