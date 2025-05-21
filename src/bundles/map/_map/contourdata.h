// vi: set expandtab shiftwidth=4 softtabstop=4:

/*
 * === UCSF ChimeraX Copyright ===
 * Copyright 2022 Regents of the University of California. All rights reserved.
 * The ChimeraX application is provided pursuant to the ChimeraX license
 * agreement, which covers academic and commercial uses. For more details, see
 * <https://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
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
//
#ifndef CONTOURDATA_HEADER_INCLUDED
#define CONTOURDATA_HEADER_INCLUDED

// ----------------------------------------------------------------------------
// A cube has 8 vertices and 12 edges.  These are numbered 0 to 7 and
// 0 to 11 in a way that comes from the original marching cubes paper.
//
// Vertex positions 0 to 7:
//
//   (0,0,0), (1,0,0), (1,1,0), (0,1,0), (0,0,1), (1,0,1), (1,1,1), (0,1,1)
//
// Edges 0 - 11:
//
//   (0,1), (1,2), (2,3), (3,0), (4,5), (5,6),
//   (6,7), (7,4), (0,4), (1,5), (2,6), (3,7)
//
// Vertex and edge and face numbering diagram
//
//                7 ---- 6 ---- 6
//               /|            /|       Faces: bottom = 0, front = 1, right = 2
//              7 |           5 |              back = 3, left = 4, top = 5
//             /  11         /  10
//            4 ---- 4 ---- 5   |
//            |   |         |   |           
//            |   3 ---- 2 -|-- 2           
//            8  /          9  /            
//            | 3           | 1             
//            |/            |/            
//            0 ---- 0 ---- 1    
//
// The cube_edge_info structure defines the edge numbering.
//
#define EDGE_A00 0
#define EDGE_1A0 1
#define EDGE_A10 2
#define EDGE_0A0 3
#define EDGE_A01 4
#define EDGE_1A1 5
#define EDGE_A11 6
#define EDGE_0A1 7
#define EDGE_00A 8
#define EDGE_10A 9
#define EDGE_11A 10
#define EDGE_01A 11

#define CORNER_000 0
#define CORNER_100 1
#define CORNER_110 2
#define CORNER_010 3
#define CORNER_001 4
#define CORNER_101 5
#define CORNER_111 6
#define CORNER_011 7

typedef int Edge_Number;
typedef int Corner_Number;

struct cube_edge_info
{
  int vertex_1, vertex_2;		// 0-7
  int base_0, base_1, base_2;		// 0 or 1 values
  int axis;				// 0-2
};
extern const struct cube_edge_info cube_edges[12];

// ----------------------------------------------------------------------------
// Table of triangles for making isosurfaces.
//
// First index is 8 bit value, each bit indicating whether data value is
// above or below threshold.  The least significant bit is vertex 0 and most
// significant bit is vertex 7 using the vertex numbering defined above.
// The table values for a bit pattern are edge indices, 3 per triangle,
// up to 5 triangles worth, terminated by a -1.
//
extern int triangle_table[256][16];

// ----------------------------------------------------------------------------
// Table maps face number and corner bits to face bits used to index
// cap_triangle_table which lists triples of grid cell vertex numbers
// (0-11 for edge cuts, 12-19 for cell corners) to form capping triangles
// where contour surface reaches edge of volume data array.
//
extern int face_corner_bits[6][256];
extern int cap_triangle_table[6][16][10];

#endif
