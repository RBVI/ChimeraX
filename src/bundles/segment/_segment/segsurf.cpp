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
// Compute a contour surface using marching cubes algorithm
//
#include <math.h>		// use sqrt()
#include <Python.h>			// use PyObject
#include <map>
#include <vector>

#include <iostream>			// use std:cerr for debugging

#include <arrays/pythonarray.h>		// use array_from_python()
#include <arrays/rcarray.h>		// use call_template_function()

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

// Right hand triangle normal points opposite gradient.
int triangle_table[256][16] =  
{
{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{8, 0, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{1, 0, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{8, 1, 3, 8, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{2, 1, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{8, 0, 3, 2, 1, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{2, 9, 10, 2, 0, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{8, 2, 3, 10, 2, 8, 9, 10, 8, -1, -1, -1, -1, -1, -1, -1},
{11, 3, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{11, 0, 2, 11, 8, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{9, 1, 0, 3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{11, 1, 2, 9, 1, 11, 8, 9, 11, -1, -1, -1, -1, -1, -1, -1},
{10, 3, 1, 10, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{10, 0, 1, 8, 0, 10, 11, 8, 10, -1, -1, -1, -1, -1, -1, -1},
{9, 3, 0, 11, 3, 9, 10, 11, 9, -1, -1, -1, -1, -1, -1, -1},
{8, 9, 10, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{7, 4, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{3, 4, 0, 3, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{1, 0, 9, 4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{1, 4, 9, 7, 4, 1, 3, 7, 1, -1, -1, -1, -1, -1, -1, -1},
{2, 1, 10, 4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{4, 3, 7, 0, 3, 4, 2, 1, 10, -1, -1, -1, -1, -1, -1, -1},
{2, 9, 10, 0, 9, 2, 4, 8, 7, -1, -1, -1, -1, -1, -1, -1},
{10, 2, 9, 9, 2, 7, 7, 2, 3, 9, 7, 4, -1, -1, -1, -1},
{4, 8, 7, 11, 3, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{4, 11, 7, 2, 11, 4, 0, 2, 4, -1, -1, -1, -1, -1, -1, -1},
{0, 9, 1, 4, 8, 7, 3, 2, 11, -1, -1, -1, -1, -1, -1, -1},
{7, 4, 11, 4, 9, 11, 11, 9, 2, 2, 9, 1, -1, -1, -1, -1},
{10, 3, 1, 11, 3, 10, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1},
{11, 1, 10, 4, 1, 11, 0, 1, 4, 11, 7, 4, -1, -1, -1, -1},
{7, 4, 8, 0, 9, 11, 11, 9, 10, 0, 11, 3, -1, -1, -1, -1},
{7, 4, 11, 11, 4, 9, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1},
{5, 9, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{5, 9, 4, 8, 0, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{5, 0, 4, 5, 1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{5, 8, 4, 3, 8, 5, 1, 3, 5, -1, -1, -1, -1, -1, -1, -1},
{2, 1, 10, 5, 9, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 3, 8, 2, 1, 10, 9, 4, 5, -1, -1, -1, -1, -1, -1, -1},
{2, 5, 10, 4, 5, 2, 0, 4, 2, -1, -1, -1, -1, -1, -1, -1},
{10, 2, 5, 2, 3, 5, 5, 3, 4, 4, 3, 8, -1, -1, -1, -1},
{5, 9, 4, 3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{11, 0, 2, 8, 0, 11, 9, 4, 5, -1, -1, -1, -1, -1, -1, -1},
{5, 0, 4, 1, 0, 5, 3, 2, 11, -1, -1, -1, -1, -1, -1, -1},
{1, 2, 5, 5, 2, 8, 8, 2, 11, 8, 4, 5, -1, -1, -1, -1},
{3, 10, 11, 1, 10, 3, 5, 9, 4, -1, -1, -1, -1, -1, -1, -1},
{9, 4, 5, 8, 0, 1, 10, 8, 1, 11, 8, 10, -1, -1, -1, -1},
{4, 5, 0, 0, 5, 11, 11, 5, 10, 0, 11, 3, -1, -1, -1, -1},
{4, 5, 8, 8, 5, 10, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1},
{7, 9, 8, 7, 5, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{3, 9, 0, 5, 9, 3, 7, 5, 3, -1, -1, -1, -1, -1, -1, -1},
{7, 0, 8, 1, 0, 7, 5, 1, 7, -1, -1, -1, -1, -1, -1, -1},
{5, 1, 3, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{7, 9, 8, 5, 9, 7, 1, 10, 2, -1, -1, -1, -1, -1, -1, -1},
{1, 10, 2, 5, 9, 0, 3, 5, 0, 7, 5, 3, -1, -1, -1, -1},
{0, 8, 2, 2, 8, 5, 5, 8, 7, 5, 10, 2, -1, -1, -1, -1},
{10, 2, 5, 5, 2, 3, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1},
{9, 7, 5, 8, 7, 9, 11, 3, 2, -1, -1, -1, -1, -1, -1, -1},
{5, 9, 7, 7, 9, 2, 2, 9, 0, 7, 2, 11, -1, -1, -1, -1},
{3, 2, 11, 1, 0, 8, 7, 1, 8, 5, 1, 7, -1, -1, -1, -1},
{2, 11, 1, 1, 11, 7, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1},
{5, 9, 8, 5, 8, 7, 1, 10, 3, 3, 10, 11, -1, -1, -1, -1},
{7, 5, 0, 0, 5, 9, 11, 7, 0, 0, 1, 10, 10, 11, 0, -1},
{10, 11, 0, 0, 11, 3, 5, 10, 0, 0, 8, 7, 7, 5, 0, -1},
{10, 11, 5, 11, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{6, 10, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{8, 0, 3, 10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 9, 1, 10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{8, 1, 3, 9, 1, 8, 10, 5, 6, -1, -1, -1, -1, -1, -1, -1},
{6, 1, 5, 6, 2, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{6, 1, 5, 2, 1, 6, 0, 3, 8, -1, -1, -1, -1, -1, -1, -1},
{6, 9, 5, 0, 9, 6, 2, 0, 6, -1, -1, -1, -1, -1, -1, -1},
{9, 5, 8, 8, 5, 2, 2, 5, 6, 2, 3, 8, -1, -1, -1, -1},
{3, 2, 11, 6, 10, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 11, 8, 2, 11, 0, 6, 10, 5, -1, -1, -1, -1, -1, -1, -1},
{1, 0, 9, 3, 2, 11, 10, 5, 6, -1, -1, -1, -1, -1, -1, -1},
{10, 5, 6, 9, 1, 2, 11, 9, 2, 8, 9, 11, -1, -1, -1, -1},
{3, 6, 11, 5, 6, 3, 1, 5, 3, -1, -1, -1, -1, -1, -1, -1},
{8, 0, 11, 11, 0, 5, 5, 0, 1, 11, 5, 6, -1, -1, -1, -1},
{11, 3, 6, 3, 0, 6, 6, 0, 5, 5, 0, 9, -1, -1, -1, -1},
{5, 6, 9, 9, 6, 11, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1},
{10, 5, 6, 7, 4, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{3, 4, 0, 7, 4, 3, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1},
{9, 1, 0, 10, 5, 6, 4, 8, 7, -1, -1, -1, -1, -1, -1, -1},
{6, 10, 5, 9, 1, 7, 7, 1, 3, 9, 7, 4, -1, -1, -1, -1},
{1, 6, 2, 5, 6, 1, 7, 4, 8, -1, -1, -1, -1, -1, -1, -1},
{2, 1, 5, 2, 5, 6, 0, 3, 4, 4, 3, 7, -1, -1, -1, -1},
{4, 8, 7, 0, 9, 5, 6, 0, 5, 2, 0, 6, -1, -1, -1, -1},
{3, 7, 9, 9, 7, 4, 2, 3, 9, 9, 5, 6, 6, 2, 9, -1},
{11, 3, 2, 8, 7, 4, 6, 10, 5, -1, -1, -1, -1, -1, -1, -1},
{10, 5, 6, 7, 4, 2, 2, 4, 0, 7, 2, 11, -1, -1, -1, -1},
{1, 0, 9, 7, 4, 8, 3, 2, 11, 10, 5, 6, -1, -1, -1, -1},
{2, 9, 1, 11, 9, 2, 4, 9, 11, 11, 7, 4, 10, 5, 6, -1},
{4, 8, 7, 11, 3, 5, 5, 3, 1, 11, 5, 6, -1, -1, -1, -1},
{1, 5, 11, 11, 5, 6, 0, 1, 11, 11, 7, 4, 4, 0, 11, -1},
{5, 0, 9, 6, 0, 5, 3, 0, 6, 6, 11, 3, 4, 8, 7, -1},
{5, 6, 9, 9, 6, 11, 7, 4, 9, 11, 7, 9, -1, -1, -1, -1},
{4, 10, 9, 4, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{10, 4, 6, 9, 4, 10, 8, 0, 3, -1, -1, -1, -1, -1, -1, -1},
{0, 10, 1, 6, 10, 0, 4, 6, 0, -1, -1, -1, -1, -1, -1, -1},
{3, 8, 1, 1, 8, 6, 6, 8, 4, 1, 6, 10, -1, -1, -1, -1},
{4, 1, 9, 2, 1, 4, 6, 2, 4, -1, -1, -1, -1, -1, -1, -1},
{0, 3, 8, 2, 1, 9, 4, 2, 9, 6, 2, 4, -1, -1, -1, -1},
{2, 0, 4, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{3, 8, 2, 2, 8, 4, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1},
{4, 10, 9, 6, 10, 4, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1},
{8, 0, 2, 8, 2, 11, 9, 4, 10, 10, 4, 6, -1, -1, -1, -1},
{11, 3, 2, 1, 0, 6, 6, 0, 4, 1, 6, 10, -1, -1, -1, -1},
{4, 6, 1, 1, 6, 10, 8, 4, 1, 1, 2, 11, 11, 8, 1, -1},
{6, 9, 4, 3, 9, 6, 1, 9, 3, 6, 11, 3, -1, -1, -1, -1},
{11, 8, 1, 1, 8, 0, 6, 11, 1, 1, 9, 4, 4, 6, 1, -1},
{11, 3, 6, 6, 3, 0, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1},
{4, 6, 8, 6, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{10, 7, 6, 8, 7, 10, 9, 8, 10, -1, -1, -1, -1, -1, -1, -1},
{7, 0, 3, 10, 0, 7, 9, 0, 10, 7, 6, 10, -1, -1, -1, -1},
{6, 10, 7, 10, 1, 7, 7, 1, 8, 8, 1, 0, -1, -1, -1, -1},
{6, 10, 7, 7, 10, 1, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1},
{2, 1, 6, 6, 1, 8, 8, 1, 9, 6, 8, 7, -1, -1, -1, -1},
{6, 2, 9, 9, 2, 1, 7, 6, 9, 9, 0, 3, 3, 7, 9, -1},
{8, 7, 0, 0, 7, 6, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1},
{3, 7, 2, 7, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{3, 2, 11, 6, 10, 8, 8, 10, 9, 6, 8, 7, -1, -1, -1, -1},
{0, 2, 7, 7, 2, 11, 9, 0, 7, 7, 6, 10, 10, 9, 7, -1},
{8, 1, 0, 7, 1, 8, 10, 1, 7, 7, 6, 10, 3, 2, 11, -1},
{2, 11, 1, 1, 11, 7, 6, 10, 1, 7, 6, 1, -1, -1, -1, -1},
{9, 8, 6, 6, 8, 7, 1, 9, 6, 6, 11, 3, 3, 1, 6, -1},
{9, 0, 1, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{8, 7, 0, 0, 7, 6, 11, 3, 0, 6, 11, 0, -1, -1, -1, -1},
{11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{6, 7, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 3, 8, 7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{1, 0, 9, 7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{1, 8, 9, 3, 8, 1, 7, 11, 6, -1, -1, -1, -1, -1, -1, -1},
{1, 10, 2, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{2, 1, 10, 0, 3, 8, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1},
{9, 2, 0, 10, 2, 9, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1},
{11, 6, 7, 10, 2, 3, 8, 10, 3, 9, 10, 8, -1, -1, -1, -1},
{2, 7, 3, 2, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 7, 8, 6, 7, 0, 2, 6, 0, -1, -1, -1, -1, -1, -1, -1},
{7, 2, 6, 3, 2, 7, 1, 0, 9, -1, -1, -1, -1, -1, -1, -1},
{6, 1, 2, 8, 1, 6, 9, 1, 8, 7, 8, 6, -1, -1, -1, -1},
{7, 10, 6, 1, 10, 7, 3, 1, 7, -1, -1, -1, -1, -1, -1, -1},
{7, 10, 6, 7, 1, 10, 8, 1, 7, 0, 1, 8, -1, -1, -1, -1},
{3, 0, 7, 7, 0, 10, 10, 0, 9, 10, 6, 7, -1, -1, -1, -1},
{6, 7, 10, 10, 7, 8, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1},
{8, 6, 4, 8, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{6, 3, 11, 0, 3, 6, 4, 0, 6, -1, -1, -1, -1, -1, -1, -1},
{6, 8, 11, 4, 8, 6, 0, 9, 1, -1, -1, -1, -1, -1, -1, -1},
{4, 9, 6, 6, 9, 3, 3, 9, 1, 3, 11, 6, -1, -1, -1, -1},
{8, 6, 4, 11, 6, 8, 10, 2, 1, -1, -1, -1, -1, -1, -1, -1},
{2, 1, 10, 0, 3, 11, 6, 0, 11, 4, 0, 6, -1, -1, -1, -1},
{11, 4, 8, 6, 4, 11, 2, 0, 9, 10, 2, 9, -1, -1, -1, -1},
{9, 10, 3, 3, 10, 2, 4, 9, 3, 3, 11, 6, 6, 4, 3, -1},
{2, 8, 3, 4, 8, 2, 6, 4, 2, -1, -1, -1, -1, -1, -1, -1},
{4, 0, 2, 6, 4, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{9, 1, 0, 3, 2, 4, 4, 2, 6, 3, 4, 8, -1, -1, -1, -1},
{9, 1, 4, 4, 1, 2, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1},
{1, 8, 3, 6, 8, 1, 4, 8, 6, 10, 6, 1, -1, -1, -1, -1},
{1, 10, 0, 0, 10, 6, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1},
{6, 4, 3, 3, 4, 8, 10, 6, 3, 3, 0, 9, 9, 10, 3, -1},
{9, 10, 4, 10, 6, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{9, 4, 5, 6, 7, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{8, 0, 3, 9, 4, 5, 7, 11, 6, -1, -1, -1, -1, -1, -1, -1},
{0, 5, 1, 4, 5, 0, 6, 7, 11, -1, -1, -1, -1, -1, -1, -1},
{7, 11, 6, 3, 8, 4, 5, 3, 4, 1, 3, 5, -1, -1, -1, -1},
{5, 9, 4, 1, 10, 2, 6, 7, 11, -1, -1, -1, -1, -1, -1, -1},
{11, 6, 7, 2, 1, 10, 8, 0, 3, 9, 4, 5, -1, -1, -1, -1},
{6, 7, 11, 4, 5, 10, 2, 4, 10, 0, 4, 2, -1, -1, -1, -1},
{4, 3, 8, 5, 3, 4, 2, 3, 5, 5, 10, 2, 7, 11, 6, -1},
{2, 7, 3, 6, 7, 2, 4, 5, 9, -1, -1, -1, -1, -1, -1, -1},
{5, 9, 4, 8, 0, 6, 6, 0, 2, 8, 6, 7, -1, -1, -1, -1},
{6, 3, 2, 7, 3, 6, 5, 1, 0, 4, 5, 0, -1, -1, -1, -1},
{2, 6, 8, 8, 6, 7, 1, 2, 8, 8, 4, 5, 5, 1, 8, -1},
{5, 9, 4, 1, 10, 6, 7, 1, 6, 3, 1, 7, -1, -1, -1, -1},
{6, 1, 10, 7, 1, 6, 0, 1, 7, 7, 8, 0, 5, 9, 4, -1},
{0, 4, 10, 10, 4, 5, 3, 0, 10, 10, 6, 7, 7, 3, 10, -1},
{6, 7, 10, 10, 7, 8, 4, 5, 10, 8, 4, 10, -1, -1, -1, -1},
{9, 6, 5, 11, 6, 9, 8, 11, 9, -1, -1, -1, -1, -1, -1, -1},
{6, 3, 11, 6, 0, 3, 5, 0, 6, 9, 0, 5, -1, -1, -1, -1},
{11, 0, 8, 5, 0, 11, 1, 0, 5, 6, 5, 11, -1, -1, -1, -1},
{11, 6, 3, 3, 6, 5, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1},
{2, 1, 10, 5, 9, 11, 11, 9, 8, 5, 11, 6, -1, -1, -1, -1},
{11, 0, 3, 6, 0, 11, 9, 0, 6, 6, 5, 9, 2, 1, 10, -1},
{8, 11, 5, 5, 11, 6, 0, 8, 5, 5, 10, 2, 2, 0, 5, -1},
{11, 6, 3, 3, 6, 5, 10, 2, 3, 5, 10, 3, -1, -1, -1, -1},
{8, 5, 9, 2, 5, 8, 6, 5, 2, 8, 3, 2, -1, -1, -1, -1},
{5, 9, 6, 6, 9, 0, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1},
{5, 1, 8, 8, 1, 0, 6, 5, 8, 8, 3, 2, 2, 6, 8, -1},
{5, 1, 6, 1, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{3, 1, 6, 6, 1, 10, 8, 3, 6, 6, 5, 9, 9, 8, 6, -1},
{1, 10, 0, 0, 10, 6, 5, 9, 0, 6, 5, 0, -1, -1, -1, -1},
{3, 0, 8, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{5, 11, 10, 5, 7, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{5, 11, 10, 7, 11, 5, 3, 8, 0, -1, -1, -1, -1, -1, -1, -1},
{11, 5, 7, 10, 5, 11, 9, 1, 0, -1, -1, -1, -1, -1, -1, -1},
{7, 10, 5, 11, 10, 7, 8, 9, 1, 3, 8, 1, -1, -1, -1, -1},
{1, 11, 2, 7, 11, 1, 5, 7, 1, -1, -1, -1, -1, -1, -1, -1},
{8, 0, 3, 2, 1, 7, 7, 1, 5, 2, 7, 11, -1, -1, -1, -1},
{7, 9, 5, 2, 9, 7, 0, 9, 2, 11, 2, 7, -1, -1, -1, -1},
{5, 7, 2, 2, 7, 11, 9, 5, 2, 2, 3, 8, 8, 9, 2, -1},
{5, 2, 10, 3, 2, 5, 7, 3, 5, -1, -1, -1, -1, -1, -1, -1},
{2, 8, 0, 5, 8, 2, 7, 8, 5, 2, 10, 5, -1, -1, -1, -1},
{0, 9, 1, 10, 5, 3, 3, 5, 7, 10, 3, 2, -1, -1, -1, -1},
{8, 9, 2, 2, 9, 1, 7, 8, 2, 2, 10, 5, 5, 7, 2, -1},
{3, 1, 5, 7, 3, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{8, 0, 7, 7, 0, 1, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1},
{0, 9, 3, 3, 9, 5, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1},
{8, 9, 7, 9, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{8, 5, 4, 10, 5, 8, 11, 10, 8, -1, -1, -1, -1, -1, -1, -1},
{0, 5, 4, 11, 5, 0, 10, 5, 11, 3, 11, 0, -1, -1, -1, -1},
{1, 0, 9, 4, 8, 10, 10, 8, 11, 4, 10, 5, -1, -1, -1, -1},
{11, 10, 4, 4, 10, 5, 3, 11, 4, 4, 9, 1, 1, 3, 4, -1},
{5, 2, 1, 8, 2, 5, 11, 2, 8, 5, 4, 8, -1, -1, -1, -1},
{4, 0, 11, 11, 0, 3, 5, 4, 11, 11, 2, 1, 1, 5, 11, -1},
{2, 0, 5, 5, 0, 9, 11, 2, 5, 5, 4, 8, 8, 11, 5, -1},
{4, 9, 5, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{5, 2, 10, 5, 3, 2, 4, 3, 5, 8, 3, 4, -1, -1, -1, -1},
{10, 5, 2, 2, 5, 4, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1},
{10, 3, 2, 5, 3, 10, 8, 3, 5, 5, 4, 8, 1, 0, 9, -1},
{10, 5, 2, 2, 5, 4, 9, 1, 2, 4, 9, 2, -1, -1, -1, -1},
{4, 8, 5, 5, 8, 3, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1},
{4, 0, 5, 0, 1, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{4, 8, 5, 5, 8, 3, 0, 9, 5, 3, 0, 5, -1, -1, -1, -1},
{4, 9, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{11, 4, 7, 9, 4, 11, 10, 9, 11, -1, -1, -1, -1, -1, -1, -1},
{8, 0, 3, 9, 4, 7, 11, 9, 7, 10, 9, 11, -1, -1, -1, -1},
{10, 1, 11, 11, 1, 4, 4, 1, 0, 4, 7, 11, -1, -1, -1, -1},
{1, 3, 4, 4, 3, 8, 10, 1, 4, 4, 7, 11, 11, 10, 4, -1},
{11, 4, 7, 11, 9, 4, 2, 9, 11, 1, 9, 2, -1, -1, -1, -1},
{7, 9, 4, 11, 9, 7, 1, 9, 11, 11, 2, 1, 8, 0, 3, -1},
{7, 11, 4, 4, 11, 2, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1},
{7, 11, 4, 4, 11, 2, 3, 8, 4, 2, 3, 4, -1, -1, -1, -1},
{9, 2, 10, 7, 2, 9, 3, 2, 7, 4, 7, 9, -1, -1, -1, -1},
{10, 9, 7, 7, 9, 4, 2, 10, 7, 7, 8, 0, 0, 2, 7, -1},
{7, 3, 10, 10, 3, 2, 4, 7, 10, 10, 1, 0, 0, 4, 10, -1},
{10, 1, 2, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{9, 4, 1, 1, 4, 7, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1},
{9, 4, 1, 1, 4, 7, 8, 0, 1, 7, 8, 1, -1, -1, -1, -1},
{0, 4, 3, 4, 7, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{10, 9, 8, 11, 10, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 3, 9, 9, 3, 11, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1},
{1, 0, 10, 10, 0, 8, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1},
{1, 3, 10, 3, 11, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{2, 1, 11, 11, 1, 9, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1},
{0, 3, 9, 9, 3, 11, 2, 1, 9, 11, 2, 9, -1, -1, -1, -1},
{2, 0, 11, 0, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{3, 2, 8, 8, 2, 10, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1},
{10, 9, 2, 9, 0, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{3, 2, 8, 8, 2, 10, 1, 0, 8, 10, 1, 8, -1, -1, -1, -1},
{10, 1, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{3, 1, 8, 1, 9, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{9, 0, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{3, 0, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
};

int face_corner_numbers[6][4] = {{0,1,2,3}, {0,4,5,1}, {1,5,6,2},
				 {2,6,7,3}, {3,7,4,0}, {4,7,6,5}};

int face_edge_numbers[6][4] = {{0,1,2,3}, {8,4,9,0}, {9,5,10,1},
			       {10,6,11,2}, {11,7,8,3}, {7,6,5,4}};

int face_corner_bits[6][256] = {
{
0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
},
{
0,1,8,9,0,1,8,9,0,1,8,9,0,1,8,9,
2,3,10,11,2,3,10,11,2,3,10,11,2,3,10,11,
4,5,12,13,4,5,12,13,4,5,12,13,4,5,12,13,
6,7,14,15,6,7,14,15,6,7,14,15,6,7,14,15,
0,1,8,9,0,1,8,9,0,1,8,9,0,1,8,9,
2,3,10,11,2,3,10,11,2,3,10,11,2,3,10,11,
4,5,12,13,4,5,12,13,4,5,12,13,4,5,12,13,
6,7,14,15,6,7,14,15,6,7,14,15,6,7,14,15,
0,1,8,9,0,1,8,9,0,1,8,9,0,1,8,9,
2,3,10,11,2,3,10,11,2,3,10,11,2,3,10,11,
4,5,12,13,4,5,12,13,4,5,12,13,4,5,12,13,
6,7,14,15,6,7,14,15,6,7,14,15,6,7,14,15,
0,1,8,9,0,1,8,9,0,1,8,9,0,1,8,9,
2,3,10,11,2,3,10,11,2,3,10,11,2,3,10,11,
4,5,12,13,4,5,12,13,4,5,12,13,4,5,12,13,
6,7,14,15,6,7,14,15,6,7,14,15,6,7,14,15
},
{
0,0,1,1,8,8,9,9,0,0,1,1,8,8,9,9,
0,0,1,1,8,8,9,9,0,0,1,1,8,8,9,9,
2,2,3,3,10,10,11,11,2,2,3,3,10,10,11,11,
2,2,3,3,10,10,11,11,2,2,3,3,10,10,11,11,
4,4,5,5,12,12,13,13,4,4,5,5,12,12,13,13,
4,4,5,5,12,12,13,13,4,4,5,5,12,12,13,13,
6,6,7,7,14,14,15,15,6,6,7,7,14,14,15,15,
6,6,7,7,14,14,15,15,6,6,7,7,14,14,15,15,
0,0,1,1,8,8,9,9,0,0,1,1,8,8,9,9,
0,0,1,1,8,8,9,9,0,0,1,1,8,8,9,9,
2,2,3,3,10,10,11,11,2,2,3,3,10,10,11,11,
2,2,3,3,10,10,11,11,2,2,3,3,10,10,11,11,
4,4,5,5,12,12,13,13,4,4,5,5,12,12,13,13,
4,4,5,5,12,12,13,13,4,4,5,5,12,12,13,13,
6,6,7,7,14,14,15,15,6,6,7,7,14,14,15,15,
6,6,7,7,14,14,15,15,6,6,7,7,14,14,15,15
},
{
0,0,0,0,1,1,1,1,8,8,8,8,9,9,9,9,
0,0,0,0,1,1,1,1,8,8,8,8,9,9,9,9,
0,0,0,0,1,1,1,1,8,8,8,8,9,9,9,9,
0,0,0,0,1,1,1,1,8,8,8,8,9,9,9,9,
2,2,2,2,3,3,3,3,10,10,10,10,11,11,11,11,
2,2,2,2,3,3,3,3,10,10,10,10,11,11,11,11,
2,2,2,2,3,3,3,3,10,10,10,10,11,11,11,11,
2,2,2,2,3,3,3,3,10,10,10,10,11,11,11,11,
4,4,4,4,5,5,5,5,12,12,12,12,13,13,13,13,
4,4,4,4,5,5,5,5,12,12,12,12,13,13,13,13,
4,4,4,4,5,5,5,5,12,12,12,12,13,13,13,13,
4,4,4,4,5,5,5,5,12,12,12,12,13,13,13,13,
6,6,6,6,7,7,7,7,14,14,14,14,15,15,15,15,
6,6,6,6,7,7,7,7,14,14,14,14,15,15,15,15,
6,6,6,6,7,7,7,7,14,14,14,14,15,15,15,15,
6,6,6,6,7,7,7,7,14,14,14,14,15,15,15,15
},
{
0,8,0,8,0,8,0,8,1,9,1,9,1,9,1,9,
4,12,4,12,4,12,4,12,5,13,5,13,5,13,5,13,
0,8,0,8,0,8,0,8,1,9,1,9,1,9,1,9,
4,12,4,12,4,12,4,12,5,13,5,13,5,13,5,13,
0,8,0,8,0,8,0,8,1,9,1,9,1,9,1,9,
4,12,4,12,4,12,4,12,5,13,5,13,5,13,5,13,
0,8,0,8,0,8,0,8,1,9,1,9,1,9,1,9,
4,12,4,12,4,12,4,12,5,13,5,13,5,13,5,13,
2,10,2,10,2,10,2,10,3,11,3,11,3,11,3,11,
6,14,6,14,6,14,6,14,7,15,7,15,7,15,7,15,
2,10,2,10,2,10,2,10,3,11,3,11,3,11,3,11,
6,14,6,14,6,14,6,14,7,15,7,15,7,15,7,15,
2,10,2,10,2,10,2,10,3,11,3,11,3,11,3,11,
6,14,6,14,6,14,6,14,7,15,7,15,7,15,7,15,
2,10,2,10,2,10,2,10,3,11,3,11,3,11,3,11,
6,14,6,14,6,14,6,14,7,15,7,15,7,15,7,15
},
{
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,
9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,
4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,
13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,
2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,
10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,
11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,
6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,
15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15
}
};

int cap_triangle_table[6][16][10] = {
{
{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 12, 3, -1, -1, -1, -1, -1, -1, -1},
{1, 13, 0, -1, -1, -1, -1, -1, -1, -1},
{13, 12, 1, 1, 12, 3, -1, -1, -1, -1},
{2, 14, 1, -1, -1, -1, -1, -1, -1, -1},
{0, 12, 3, 2, 14, 1, -1, -1, -1, -1},
{14, 13, 2, 2, 13, 0, -1, -1, -1, -1},
{3, 13, 12, 2, 13, 3, 14, 13, 2, -1},
{3, 15, 2, -1, -1, -1, -1, -1, -1, -1},
{12, 15, 0, 0, 15, 2, -1, -1, -1, -1},
{1, 13, 0, 3, 15, 2, -1, -1, -1, -1},
{13, 12, 1, 1, 12, 2, 2, 12, 15, -1},
{15, 14, 3, 3, 14, 1, -1, -1, -1, -1},
{12, 15, 0, 0, 15, 1, 1, 15, 14, -1},
{0, 14, 13, 3, 14, 0, 15, 14, 3, -1},
{13, 12, 14, 14, 12, 15, -1, -1, -1, -1},
},
{
{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{8, 12, 0, -1, -1, -1, -1, -1, -1, -1},
{4, 16, 8, -1, -1, -1, -1, -1, -1, -1},
{16, 12, 4, 4, 12, 0, -1, -1, -1, -1},
{9, 17, 4, -1, -1, -1, -1, -1, -1, -1},
{8, 12, 0, 9, 17, 4, -1, -1, -1, -1},
{17, 16, 9, 9, 16, 8, -1, -1, -1, -1},
{0, 16, 12, 9, 16, 0, 17, 16, 9, -1},
{0, 13, 9, -1, -1, -1, -1, -1, -1, -1},
{12, 13, 8, 8, 13, 9, -1, -1, -1, -1},
{4, 16, 8, 0, 13, 9, -1, -1, -1, -1},
{16, 12, 4, 4, 12, 9, 9, 12, 13, -1},
{13, 17, 0, 0, 17, 4, -1, -1, -1, -1},
{12, 13, 8, 8, 13, 4, 4, 13, 17, -1},
{8, 17, 16, 0, 17, 8, 13, 17, 0, -1},
{16, 12, 17, 17, 12, 13, -1, -1, -1, -1},
},
{
{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{9, 13, 1, -1, -1, -1, -1, -1, -1, -1},
{5, 17, 9, -1, -1, -1, -1, -1, -1, -1},
{17, 13, 5, 5, 13, 1, -1, -1, -1, -1},
{10, 18, 5, -1, -1, -1, -1, -1, -1, -1},
{9, 13, 1, 10, 18, 5, -1, -1, -1, -1},
{18, 17, 10, 10, 17, 9, -1, -1, -1, -1},
{1, 17, 13, 10, 17, 1, 18, 17, 10, -1},
{1, 14, 10, -1, -1, -1, -1, -1, -1, -1},
{13, 14, 9, 9, 14, 10, -1, -1, -1, -1},
{5, 17, 9, 1, 14, 10, -1, -1, -1, -1},
{17, 13, 5, 5, 13, 10, 10, 13, 14, -1},
{14, 18, 1, 1, 18, 5, -1, -1, -1, -1},
{13, 14, 9, 9, 14, 5, 5, 14, 18, -1},
{9, 18, 17, 1, 18, 9, 14, 18, 1, -1},
{17, 13, 18, 18, 13, 14, -1, -1, -1, -1},
},
{
{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{10, 14, 2, -1, -1, -1, -1, -1, -1, -1},
{6, 18, 10, -1, -1, -1, -1, -1, -1, -1},
{18, 14, 6, 6, 14, 2, -1, -1, -1, -1},
{11, 19, 6, -1, -1, -1, -1, -1, -1, -1},
{10, 14, 2, 11, 19, 6, -1, -1, -1, -1},
{19, 18, 11, 11, 18, 10, -1, -1, -1, -1},
{2, 18, 14, 11, 18, 2, 19, 18, 11, -1},
{2, 15, 11, -1, -1, -1, -1, -1, -1, -1},
{14, 15, 10, 10, 15, 11, -1, -1, -1, -1},
{6, 18, 10, 2, 15, 11, -1, -1, -1, -1},
{18, 14, 6, 6, 14, 11, 11, 14, 15, -1},
{15, 19, 2, 2, 19, 6, -1, -1, -1, -1},
{14, 15, 10, 10, 15, 6, 6, 15, 19, -1},
{10, 19, 18, 2, 19, 10, 15, 19, 2, -1},
{18, 14, 19, 19, 14, 15, -1, -1, -1, -1},
},
{
{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{11, 15, 3, -1, -1, -1, -1, -1, -1, -1},
{7, 19, 11, -1, -1, -1, -1, -1, -1, -1},
{19, 15, 7, 7, 15, 3, -1, -1, -1, -1},
{8, 16, 7, -1, -1, -1, -1, -1, -1, -1},
{11, 15, 3, 8, 16, 7, -1, -1, -1, -1},
{16, 19, 8, 8, 19, 11, -1, -1, -1, -1},
{3, 19, 15, 8, 19, 3, 16, 19, 8, -1},
{3, 12, 8, -1, -1, -1, -1, -1, -1, -1},
{15, 12, 11, 11, 12, 8, -1, -1, -1, -1},
{7, 19, 11, 3, 12, 8, -1, -1, -1, -1},
{19, 15, 7, 7, 15, 8, 8, 15, 12, -1},
{12, 16, 3, 3, 16, 7, -1, -1, -1, -1},
{15, 12, 11, 11, 12, 7, 7, 12, 16, -1},
{11, 16, 19, 3, 16, 11, 12, 16, 3, -1},
{19, 15, 16, 16, 15, 12, -1, -1, -1, -1},
},
{
{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{7, 16, 4, -1, -1, -1, -1, -1, -1, -1},
{6, 19, 7, -1, -1, -1, -1, -1, -1, -1},
{19, 16, 6, 6, 16, 4, -1, -1, -1, -1},
{5, 18, 6, -1, -1, -1, -1, -1, -1, -1},
{7, 16, 4, 5, 18, 6, -1, -1, -1, -1},
{18, 19, 5, 5, 19, 7, -1, -1, -1, -1},
{4, 19, 16, 5, 19, 4, 18, 19, 5, -1},
{4, 17, 5, -1, -1, -1, -1, -1, -1, -1},
{16, 17, 7, 7, 17, 5, -1, -1, -1, -1},
{6, 19, 7, 4, 17, 5, -1, -1, -1, -1},
{19, 16, 6, 6, 16, 5, 5, 16, 17, -1},
{17, 18, 4, 4, 18, 6, -1, -1, -1, -1},
{16, 17, 7, 7, 17, 6, 6, 17, 18, -1},
{7, 18, 19, 4, 18, 7, 17, 18, 4, -1},
{19, 16, 18, 18, 16, 17, -1, -1, -1, -1},
},
};

//
// The data values can be any of the standard C numeric types.
//
// The grid size array is in x, y, z order.
//
// The grid value for index (i0,i1,i2) where 0 <= ik < size[k] is
//
//	grid[i0*stride[0] + i1*stride[1] + i2*stride[2]]
//
// Returned vertex and triangle arrays should be freed with free_surface().
//

#include <cstdint>		// Use std::int64_t
typedef unsigned int Index;	// grid and edge indices, and surface vertex indices

typedef std::int64_t Stride;	// Array strides and pointer offsets, signed

typedef int Region_Id;

class Region_Surface
{
public:
  Region_Surface(Region_Id region_id) : region_id(region_id) {}
  Region_Id region_id;
  std::vector<float> vertices;
  std::vector<Index> triangles;
};
typedef std::vector<Region_Surface> Region_Surfaces;

class Contour_Surface
{
 public:
  virtual ~Contour_Surface() {};
  virtual void compute_surface() = 0;
  virtual void compute_surfaces() = 0;
  virtual const Region_Surfaces &surfaces() const = 0;
};

template <class Data_Type>
Contour_Surface *surface(const Data_Type *grid,
			 const Index size[3], const Stride stride[3],
			 int value, bool cap_faces);

const Index no_vertex = ~(Index)0;

// ----------------------------------------------------------------------------
// A cell is a cube in the 3D data array with corners at 8 grid points.
// Grid_Cell records the vertex numbers on cube edges and at corners needed
// for triangulating the surface within the cell including triangulating
// boundary faces of the 3D array.
//
class Grid_Cell
{
public:
  Index k0, k1;		// Cell position in xy plane.
  Index vertex[20];	// Vertex numbers for 12 edges and 8 corners.
  bool boundary;	// Contour reaches boundary.
};

// ----------------------------------------------------------------------------
// 2D array of grid cells.  Each grid cell records the vertex numbers along
// the cube edges and corners needed for triangulating the surface within the cell.
//
class Grid_Cell_List
{
public:
  Grid_Cell_List(Index size0, Index size1)
  {
    this->cell_table_size0 = size0+2;	// Pad by one grid cell.
    Index cell_table_size1 = size1+2;
    Index size = cell_table_size0 * cell_table_size1;
    this->cell_count = 0;
    this->cell_base_index = 2;
    this->cell_table = new Index[size];
    for (Index i = 0 ; i < size ; ++i)
      cell_table[i] = no_cell;
    for (Index i = 0 ; i < cell_table_size0 ; ++i)
      cell_table[i] = cell_table[size-i-1] = out_of_bounds;
    for (Index i = 0 ; i < size ; i += cell_table_size0)
      cell_table[i] = cell_table[i+cell_table_size0-1] = out_of_bounds;
  }
  ~Grid_Cell_List()
    {
      delete_cells();
      delete [] cell_table;
    }
  void set_edge_vertex(Index k0, Index k1, Edge_Number e, Index v)
  {
    Grid_Cell *c = cell(k0,k1);
    if (c)
      c->vertex[e] = v;
  }
  void set_corner_vertex(Index k0, Index k1, Corner_Number corner, Index v)
  {
    Grid_Cell *c = cell(k0,k1);
    if (c)
      {
	c->vertex[12+corner] = v;
	c->boundary = true;
      }
  }
  void finished_plane()
    {
      cell_base_index += cell_count;
      cell_count = 0;
    }

  Index cell_count;		// Number of elements of cells currently in use.
  std::vector<Grid_Cell *> cells;

private:
  static const Index out_of_bounds = 0;
  static const Index no_cell = 1;
  Index cell_table_size0;
  Index cell_base_index;	// Minimum valid cell index.
  Index *cell_table;		// Maps cell plane index to cell list index.

  // Get cell, initializing or allocating a new one if necessary.
  Grid_Cell *cell(Index k0, Index k1)
  {
    Index i = k0+1 + (k1+1)*cell_table_size0;
    Index c = cell_table[i];
    if (c == out_of_bounds)
      return NULL;

    Grid_Cell *cp;
    if (c != no_cell && c >= cell_base_index)
      cp = cells[c-cell_base_index];
    else
      {
	cell_table[i] = cell_base_index + cell_count;
	if (cell_count < cells.size())
	  cp = cells[cell_count];
	else
	  cells.push_back(cp = new Grid_Cell);
	cp->k0 = k0; cp->k1 = k1; cp->boundary = false;
	cell_count += 1;
      }
    return cp;
  }

  void delete_cells()
  {
    Index cc = cells.size();
    for (Index c = 0 ; c < cc ; ++c)
      delete cells[c];
  }
};

// ----------------------------------------------------------------------------
//
template <class Data_Type>
class Region_Test
{
public:
  Region_Test(int region_id, const int *region_ids = NULL)
    : all_regions(false), region_id(region_id), region_ids(region_ids) {}
  Region_Test(const int *region_ids = NULL)
    : all_regions(true), region_id(0), region_ids(region_ids) {}
  Region_Id operator()(Data_Type value) const
  {
    //    return region_ids[(int)value];
    if (region_ids == NULL)
      {
	if (all_regions)
	  return (Region_Id) value;
	else if ((int)value == region_id)
	  return region_id;
      }
    else
      {
	Region_Id rid = region_ids[(int)value];
	if (all_regions)
	  return rid;
	else if (rid == region_id)
	  return region_id;
      }
    return 0;
  }
  bool all() const { return all_regions; }
  const int *groups() const { return region_ids; }
private:
  const bool all_regions;
  const int region_id;
  const int * const region_ids;
};

// ----------------------------------------------------------------------------
//
class Region_Point
{
public:
  Region_Point(unsigned short int i0, unsigned short int i1, unsigned short int i2,
	       unsigned char neighbors, unsigned char boundary)
    : i0(i0), i1(i1), i2(i2), neighbors(neighbors), boundary(boundary) {}
  unsigned short int i0, i1, i2;
  unsigned char neighbors, boundary;  // 6 bits for -x,+x,-y,+y,-z,+z directions surface cut or box boundary.
};

typedef std::vector<Region_Point> Region_Points;
typedef std::map<Region_Id, Region_Points> Points_Per_Region;

// ----------------------------------------------------------------------------
//
template <class Data_Type>
class CSurface : public Contour_Surface
{
public:
  CSurface(const Data_Type *grid, const Index size[3], const Stride stride[3],
	   const Region_Test<Data_Type> &inside, bool cap_faces)
    : grid(grid), inside(inside), cap_faces(cap_faces)
    {
      for (int a = 0 ; a < 3 ; ++a)
	{ this->size[a] = size[a]; this->stride[a] = stride[a]; }
    }
  virtual ~CSurface() {}

  virtual void compute_surface();
  virtual void compute_surfaces();

  virtual const Region_Surfaces &surfaces() const { return surfs; }

private:
  const Data_Type *grid;
  Index size[3];
  Stride stride[3];
  const Region_Test<Data_Type> inside;
  bool cap_faces;
  Region_Surfaces surfs;
  std::vector<float> *vxyz;
  std::vector<Index> *tvi;

  // Methods for computing single surface.
  void mark_plane_edge_cuts(Grid_Cell_List &gp0, Grid_Cell_List &gp1, Index k2);
  void mark_interior_edge_cuts(Index k1, Index k2,
			       Grid_Cell_List &gp0, Grid_Cell_List &gp1);
  void mark_boundary_edge_cuts(Index k0, Index k1, Index k2,
			       Grid_Cell_List &gp0, Grid_Cell_List &gp1);
  void make_triangles(Grid_Cell_List &gp0, Index k2, Region_Id region_id);

  // Methods for computing multiple surfaces.
  void find_region_group_points(Points_Per_Region &region_points);
  void find_region_points(Points_Per_Region &region_points);
  void mark_point_edge_cuts(const Region_Point &rp, Grid_Cell_List &gp0, Grid_Cell_List &gp1);
  void new_surface(int region_id);

  void add_vertex_axis_0(Index k0, Index k1, Index k2, float x0,
			 Grid_Cell_List &gp0, Grid_Cell_List &gp1);
  void add_vertex_axis_1(Index k0, Index k1, Index k2, float x1,
			 Grid_Cell_List &gp0, Grid_Cell_List &gp1);
  void add_vertex_axis_2(Index k0, Index k1, float x2,
			 Grid_Cell_List &gp);

  Index add_cap_vertex_l0(Index bv, Index k0, Index k1, Index k2,
			  Grid_Cell_List &gp0, Grid_Cell_List &gp1);
  Index add_cap_vertex_r0(Index bv, Index k0, Index k1, Index k2,
			  Grid_Cell_List &gp0, Grid_Cell_List &gp1);
  Index add_cap_vertex_l1(Index bv, Index k0, Index k1, Index k2,
			  Grid_Cell_List &gp0, Grid_Cell_List &gp1);
  Index add_cap_vertex_r1(Index bv, Index k0, Index k1, Index k2,
			  Grid_Cell_List &gp0, Grid_Cell_List &gp1);
  Index add_cap_vertex_l2(Index bv, Index k0, Index k1, Index k2,
			  Grid_Cell_List &gp1);
  Index add_cap_vertex_r2(Index bv, Index k0, Index k1, Index k2,
			  Grid_Cell_List &gp0);

  void make_triangles(Grid_Cell_List &gp0, Index k2);
  void add_triangle_corner(Index v) { tvi->push_back(v); }
  Index create_vertex(float x, float y, float z)
    { vxyz->push_back(x); vxyz->push_back(y); vxyz->push_back(z);
      return vxyz->size()/3-1; }
  void make_cap_triangles(int face, int bits, Index *cell_vertices)
    {
      int fbits = face_corner_bits[face][bits];
      int *t = cap_triangle_table[face][fbits];
      for (int v = *t ; v != -1 ; ++t, v = *t)
	add_triangle_corner(cell_vertices[v]);
    }
  void make_cap_triangles(Index k0, Index k1, Index k2, int bits, Index *cell_vertices);
};

// ----------------------------------------------------------------------------
// The grid value for index (i0,i1,i2) where 0 <= ik < size[k] is
//
//	grid[i0*stride[0] + i1*stride[1] + i2*stride[2]]
//
template <class Data_Type>
void CSurface<Data_Type>::compute_surface()
{
  //
  // If grid point value equals specified value check if 6 connected edges
  // cross contour surface and make vertex, add vertex to 4 bordering
  // grid cells, triangulate grid cells between two z grid planes.
  //
  new_surface(1);
  Grid_Cell_List gcp0(size[0]-1, size[1]-1), gcp1(size[0]-1, size[1]-1);
  for (Index k2 = 0 ; k2 < size[2] ; ++k2)
    {
      Grid_Cell_List &gp0 = (k2%2 ? gcp1 : gcp0), &gp1 = (k2%2 ? gcp0 : gcp1);
      mark_plane_edge_cuts(gp0, gp1, k2);

      if (k2 > 0)
	make_triangles(gp0, k2);	// Create triangles for cell plane.

      gp0.finished_plane();
    }
}

// ----------------------------------------------------------------------------
//
template <class Data_Type>
void CSurface<Data_Type>::compute_surfaces()
{
  Region_Surfaces surfs;
  Points_Per_Region region_points;
  if (inside.all() && inside.groups())
    find_region_group_points(region_points);
  else
    find_region_points(region_points);
  //  std::cerr << "find_region_points() " <<  time << std::endl;
  Grid_Cell_List gcp0(size[0]-1, size[1]-1), gcp1(size[0]-1, size[1]-1);
  for (auto rp = region_points.begin() ; rp != region_points.end() ; ++rp)
    {
      Region_Id rid = rp->first;
      new_surface(rid);
      Region_Points &rpoints = rp->second;
      Index i2 = 0;
      Grid_Cell_List *gp0 = &gcp0, *gp1 = &gcp1;
      for (auto p = rpoints.begin() ; p != rpoints.end() ; ++p)
	{
	  if (p->i2 > i2)
	    {
	      if (i2 > 0)
		make_triangles(*gp0, i2, rid);	// Create triangles for cell plane.
	      gp0->finished_plane();
	      if (p->i2 > i2+1)
		gp1->finished_plane();
	      i2 = p->i2;
	      gp0 = (i2%2 ? &gcp1 : &gcp0);
	      gp1 = (i2%2 ? &gcp0 : &gcp1);
	    }
	  mark_point_edge_cuts(*p, *gp0, *gp1);
	}
      if (i2 > 0)
	{
	  // Create triangles for last planes
	  make_triangles(*gp0, i2, rid);	
	  if (i2+1 < size[2])
	    make_triangles(*gp1, i2+1, rid);
	}
      gp0->finished_plane();
      gp1->finished_plane();
    }
}

// ----------------------------------------------------------------------------
//
template <class Data_Type>
void CSurface<Data_Type>::new_surface(int region_id)
{
  surfs.push_back(Region_Surface(region_id));
  Region_Surface &s = surfs[surfs.size()-1];
  vxyz = &s.vertices;
  tvi = &s.triangles;
}

// ----------------------------------------------------------------------------
//
template <class Data_Type>
void CSurface<Data_Type>::find_region_group_points(Points_Per_Region &region_points)
{
  // Optimize group id array lookup.
  // The inside() method is 15% slower even when inlined.

  Index k0_size = size[0], k1_size = size[1], k2_size = size[2];
  Stride step0 = stride[0], step1 = stride[1], step2 = stride[2];
  Region_Points *rpoints;
  Region_Id last_region_id = -1;
  const int *group_ids = inside.groups();
  for (Index k2 = 0 ; k2 < k2_size ; ++k2)
	for (Index k1 = 0 ; k1 < k1_size ; ++k1)
	  {
	    const Data_Type *g = grid + step2*(Stride)k2 + step1*(Stride)k1;
	    unsigned char b12 = ((k1 == 0 ? 4 : 0) | (k1+1 == k1_size ? 8 : 0) |
				 (k2 == 0 ? 16 : 0) | (k2+1 == k2_size ? 32 : 0));
	    for (Index k0 = 0 ; k0 < k0_size ; ++k0, g += step0)
	      {
		int region_id = group_ids[(int)*g];
		if (region_id)
		  {
		    //		    sum += 1 + b12;
		    unsigned char b = b12 | (k0 == 0 ? 1 : 0) | (k0+1 == k0_size ? 2 : 0);
		    unsigned char n = ((!(b & 1) && group_ids[(int)*(g-step0)] != region_id ? 1 : 0) |
				   (!(b & 2) && group_ids[(int)g[step0]] != region_id ? 2 : 0) |
				   (!(b & 4) && group_ids[(int)*(g-step1)] != region_id ? 4 : 0) |
				   (!(b & 8) && group_ids[(int)g[step1]] != region_id ? 8 : 0) |
				   (!(b & 16) && group_ids[(int)*(g-step2)] != region_id ? 16 : 0) |
				   (!(b & 32) && group_ids[(int)g[step2]] != region_id ? 32 : 0));
		    if (b || n)
		      {
			if (region_id != last_region_id)
			  {
			    // Use last region lookup when possible to avoid time consuming map lookup.
			    rpoints = &region_points[region_id];
			    last_region_id = region_id;
			  }
			rpoints->push_back(Region_Point(k0,k1,k2,n,b));
		      }
		  }
	      }
	  }
}

// ----------------------------------------------------------------------------
//
template <class Data_Type>
void CSurface<Data_Type>::find_region_points(Points_Per_Region &region_points)
{
  //  auto begin = std::chrono::high_resolution_clock::now();
  Index k0_size = size[0], k1_size = size[1], k2_size = size[2];
  Stride step0 = stride[0], step1 = stride[1], step2 = stride[2];
  Region_Points *rpoints;
  Region_Id last_region_id = -1;

  for (Index k2 = 0 ; k2 < k2_size ; ++k2)
    for (Index k1 = 0 ; k1 < k1_size ; ++k1)
	  {
	    const Data_Type *g = grid + step2*(Stride)k2 + step1*(Stride)k1;
	    unsigned char b12 = ((k1 == 0 ? 4 : 0) | (k1+1 == k1_size ? 8 : 0) |
				 (k2 == 0 ? 16 : 0) | (k2+1 == k2_size ? 32 : 0));
	    for (Index k0 = 0 ; k0 < k0_size ; ++k0, g += step0)
	      {
		int region_id = inside(*g);
		if (region_id)
		  {
		    unsigned char b = b12 | (k0 == 0 ? 1 : 0) | (k0+1 == k0_size ? 2 : 0);
		    unsigned char n = ((!(b & 1) && inside(*(g-step0)) != region_id ? 1 : 0) |
				       (!(b & 2) && inside(g[step0]) != region_id ? 2 : 0) |
				       (!(b & 4) && inside(*(g-step1)) != region_id ? 4 : 0) |
				       (!(b & 8) && inside(g[step1]) != region_id ? 8 : 0) |
				       (!(b & 16) && inside(*(g-step2)) != region_id ? 16 : 0) |
				       (!(b & 32) && inside(g[step2]) != region_id ? 32 : 0));
		    if (b || n)
		      {
			if (region_id != last_region_id)
			  {
			    // Use last region lookup when possible to avoid time consuming map lookup.
			    rpoints = &region_points[region_id];
			    last_region_id = region_id;
			  }
			rpoints->push_back(Region_Point(k0,k1,k2,n,b));
		      }
		  }
	      }
	  }

  // auto end = std::chrono::high_resolution_clock::now();
  // long t = (long)std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count();
  // std::cerr << "find_region_points() time " << t << std::endl;
}

// ----------------------------------------------------------------------------
// Compute edge cut vertices in 6 directions and capping corner vertex for
// boundary grid points.
//
template <class Data_Type>
inline void CSurface<Data_Type>::mark_point_edge_cuts(const Region_Point &rp,
						      Grid_Cell_List &gp0,
						      Grid_Cell_List &gp1)
{
  Index k0 = rp.i0, k1 = rp.i1, k2 = rp.i2;
  unsigned char n = rp.neighbors, b = rp.boundary;
  Index bv = no_vertex;

  // Axis 0 left
  if (n & 1)
    add_vertex_axis_0(k0-1, k1, k2, k0-0.5, gp0, gp1);
  else if (b & 1)
    bv = add_cap_vertex_l0(bv, k0, k1, k2, gp0, gp1);

  // Axis 0 right
  if (n & 2)
    add_vertex_axis_0(k0, k1, k2, k0+0.5, gp0, gp1);
  else if (b & 2)
    bv = add_cap_vertex_r0(bv, k0, k1, k2, gp0, gp1);

  // Axis 1 left
  if (n & 4)
    add_vertex_axis_1(k0, k1-1, k2, k1-0.5, gp0, gp1);
  else if (b & 4)
    bv = add_cap_vertex_l1(bv, k0, k1, k2, gp0, gp1);

  // Axis 1 right
  if (n & 8)
    add_vertex_axis_1(k0, k1, k2, k1+0.5, gp0, gp1);
  else if (b & 8)
    bv = add_cap_vertex_r1(bv, k0, k1, k2, gp0, gp1);

  // Axis 2 left
  if (n & 16)
    add_vertex_axis_2(k0, k1, k2-0.5, gp0);
  else if (b & 16)
    bv = add_cap_vertex_l2(bv, k0, k1, k2, gp1);

  // Axis 2 right
  if (n & 32)
    add_vertex_axis_2(k0, k1, k2+0.5, gp1);
  else if (b & 32)
    bv = add_cap_vertex_r2(bv, k0, k1, k2, gp0);
}

// ----------------------------------------------------------------------------
//
template <class Data_Type>
void CSurface<Data_Type>::mark_plane_edge_cuts(Grid_Cell_List &gp0,
					       Grid_Cell_List &gp1,
					       Index k2)
{
  Index k0_size = size[0], k1_size = size[1], k2_size = size[2];

  for (Index k1 = 0 ; k1 < k1_size ; ++k1)
    {
      if (k1 == 0 || k1+1 == k1_size || k2 == 0 || k2+1 == k2_size)
	for (Index k0 = 0 ; k0 < k0_size ; ++k0)
	  mark_boundary_edge_cuts(k0, k1, k2, gp0, gp1);
      else
	{
	  if (k0_size > 0)
	    mark_boundary_edge_cuts(0, k1, k2, gp0, gp1);

	  mark_interior_edge_cuts(k1, k2, gp0, gp1);
	  
	  if (k0_size > 1)
	    mark_boundary_edge_cuts(k0_size-1, k1, k2, gp0, gp1);
	}
    }
}

// ----------------------------------------------------------------------------
// Compute edge cut vertices in 6 directions along axis 0 not including the
// axis end points.  k1 and k2 axis values must not be on the boundary.
// This allows faster processing since boundary checking is not needed.
//
template <class Data_Type>
inline void CSurface<Data_Type>::mark_interior_edge_cuts(Index k1, Index k2,
							 Grid_Cell_List &gp0,
							 Grid_Cell_List &gp1)
{
  Stride step0 = stride[0], step1 = stride[1], step2 = stride[2];
  Index k0_max = size[0]-1;

  const Data_Type *g = grid + step2*(Stride)k2 + step1*(Stride)k1 + step0;
  for (Index k0 = 1 ; k0 < k0_max ; ++k0, g += step0)
    {
      if (inside(*g))
	{
	  // Grid point equals value.
	  // Look at 6 neigbors along x,y,z axes for values not equal to value.
	  if (!inside(*(g-step0)))
	    add_vertex_axis_0(k0-1, k1, k2, k0-0.5, gp0, gp1);
	  if (!inside(g[step0]))
	    add_vertex_axis_0(k0, k1, k2, k0+0.5, gp0, gp1);
	  if (!inside(*(g-step1)))
	    add_vertex_axis_1(k0, k1-1, k2, k1-0.5, gp0, gp1);
	  if (!inside(g[step1]))
	    add_vertex_axis_1(k0, k1, k2, k1+0.5, gp0, gp1);
	  if (!inside(*(g-step2)))
	    add_vertex_axis_2(k0, k1, k2-0.5, gp0);
	  if (!inside(g[step2]))
	    add_vertex_axis_2(k0, k1, k2+0.5, gp1);
	}
    }
}

// ----------------------------------------------------------------------------
// Compute edge cut vertices in 6 directions and capping corner vertex for
// boundary grid points.
//
template <class Data_Type>
inline void CSurface<Data_Type>::mark_boundary_edge_cuts(Index k0, Index k1, Index k2,
							 Grid_Cell_List &gp0,
							 Grid_Cell_List &gp1)
{
  Stride step0 = stride[0], step1 = stride[1], step2 = stride[2];
  Index k0_size = size[0], k1_size = size[1], k2_size = size[2];
  const Data_Type *g = grid + step2*(Stride)k2 + step1*(Stride)k1 + step0*(Stride)k0;
  if (!inside(*g))
    return;

  // Check 6 neighbor vertices for edge crossings.

  Index bv = no_vertex;

  // Axis 0 left
  if (k0 > 0)
    {
      if (!inside(*(g-step0)))
	add_vertex_axis_0(k0-1, k1, k2, k0-0.5, gp0, gp1);
    }
  else if (cap_faces)  // boundary vertex for capping box faces.
    bv = add_cap_vertex_l0(bv, k0, k1, k2, gp0, gp1);

  // Axis 0 right
  if (k0+1 < k0_size)
    {
      if (!inside(g[step0]))
	add_vertex_axis_0(k0, k1, k2, k0+0.5, gp0, gp1);
    }
  else if (cap_faces)
    bv = add_cap_vertex_r0(bv, k0, k1, k2, gp0, gp1);

  // Axis 1 left
  if (k1 > 0)
    {
      if (!inside(*(g-step1)))
	add_vertex_axis_1(k0, k1-1, k2, k1-0.5, gp0, gp1);
    }
  else if (cap_faces)
    bv = add_cap_vertex_l1(bv, k0, k1, k2, gp0, gp1);

  // Axis 1 right
  if (k1+1 < k1_size)
    {
      if (!inside(g[step1]))
	add_vertex_axis_1(k0, k1, k2, k1+0.5, gp0, gp1);
    }
  else if (cap_faces)
    bv = add_cap_vertex_r1(bv, k0, k1, k2, gp0, gp1);

  // Axis 2 left
  if (k2 > 0)
    {
      if (!inside(*(g-step2)))
	add_vertex_axis_2(k0, k1, k2-0.5, gp0);
    }
  else if (cap_faces)
    bv = add_cap_vertex_l2(bv, k0, k1, k2, gp1);

  // Axis 2 right
  if (k2+1 < k2_size)
    {
      if (!inside(g[step2]))
	add_vertex_axis_2(k0, k1, k2+0.5, gp1);
    }
  else if (cap_faces)
    bv = add_cap_vertex_r2(bv, k0, k1, k2, gp0);
}

// ----------------------------------------------------------------------------
// Add axis 0 edge cut to four adjoining grid cells.
//
template <class Data_Type>
void CSurface<Data_Type>::add_vertex_axis_0(Index k0, Index k1, Index k2, float x0,
					    Grid_Cell_List &gp0, Grid_Cell_List &gp1)
{
  Index v = create_vertex(x0,k1,k2);
  gp0.set_edge_vertex(k0, k1-1, EDGE_A11, v);
  gp0.set_edge_vertex(k0, k1, EDGE_A01, v);
  gp1.set_edge_vertex(k0, k1-1, EDGE_A10, v);
  gp1.set_edge_vertex(k0, k1, EDGE_A00, v);
}

// ----------------------------------------------------------------------------
// Add axis 1 edge cut to four adjoining grid cells.
//
template <class Data_Type>
void CSurface<Data_Type>::add_vertex_axis_1(Index k0, Index k1, Index k2, float x1,
					    Grid_Cell_List &gp0, Grid_Cell_List &gp1)
{
  Index v = create_vertex(k0,x1,k2);
  gp0.set_edge_vertex(k0-1, k1, EDGE_1A1, v);
  gp0.set_edge_vertex(k0, k1, EDGE_0A1, v);
  gp1.set_edge_vertex(k0-1, k1, EDGE_1A0, v);
  gp1.set_edge_vertex(k0, k1, EDGE_0A0, v);
}

// ----------------------------------------------------------------------------
// Add axis 2 edge cut to four adjoining grid cells.
//
template <class Data_Type>
void CSurface<Data_Type>::add_vertex_axis_2(Index k0, Index k1, float x2,
					    Grid_Cell_List &gp)
{
  Index v = create_vertex(k0,k1,x2);
  gp.set_edge_vertex(k0, k1, EDGE_00A, v);
  gp.set_edge_vertex(k0-1, k1, EDGE_10A, v);
  gp.set_edge_vertex(k0, k1-1, EDGE_01A, v);
  gp.set_edge_vertex(k0-1, k1-1, EDGE_11A, v);
}

// ----------------------------------------------------------------------------
//
template <class Data_Type>
Index CSurface<Data_Type>::add_cap_vertex_l0(Index bv,
					     Index k0, Index k1, Index k2,
					     Grid_Cell_List &gp0,
					     Grid_Cell_List &gp1)
{
  if (bv == no_vertex)
    bv = create_vertex(k0,k1,k2);
  gp0.set_corner_vertex(k0, k1-1, CORNER_011, bv);
  gp0.set_corner_vertex(k0, k1, CORNER_001, bv);
  gp1.set_corner_vertex(k0, k1-1, CORNER_010, bv);
  gp1.set_corner_vertex(k0, k1, CORNER_000, bv);
  return bv;
}
// ----------------------------------------------------------------------------
//
template <class Data_Type>
Index CSurface<Data_Type>::add_cap_vertex_r0(Index bv,
					     Index k0, Index k1, Index k2,
					     Grid_Cell_List &gp0,
					     Grid_Cell_List &gp1)
{
  if (bv == no_vertex)
    bv = create_vertex(k0,k1,k2);
  gp0.set_corner_vertex(k0-1, k1-1, CORNER_111, bv);
  gp0.set_corner_vertex(k0-1, k1, CORNER_101, bv);
  gp1.set_corner_vertex(k0-1, k1-1, CORNER_110, bv);
  gp1.set_corner_vertex(k0-1, k1, CORNER_100, bv);
  return bv;
}

// ----------------------------------------------------------------------------
//
template <class Data_Type>
Index CSurface<Data_Type>::add_cap_vertex_l1(Index bv,
					     Index k0, Index k1, Index k2,
					     Grid_Cell_List &gp0,
					     Grid_Cell_List &gp1)
{
  if (bv == no_vertex)
    bv = create_vertex(k0,k1,k2);
  gp0.set_corner_vertex(k0-1, k1, CORNER_101, bv);
  gp0.set_corner_vertex(k0, k1, CORNER_001, bv);
  gp1.set_corner_vertex(k0-1, k1, CORNER_100, bv);
  gp1.set_corner_vertex(k0, k1, CORNER_000, bv);
  return bv;
}

// ----------------------------------------------------------------------------
//
template <class Data_Type>
Index CSurface<Data_Type>::add_cap_vertex_r1(Index bv,
					     Index k0, Index k1, Index k2,
					     Grid_Cell_List &gp0,
					     Grid_Cell_List &gp1)
{
  if (bv == no_vertex)
    bv = create_vertex(k0,k1,k2);
  gp0.set_corner_vertex(k0-1, k1-1, CORNER_111, bv);
  gp0.set_corner_vertex(k0, k1-1, CORNER_011, bv);
  gp1.set_corner_vertex(k0-1, k1-1, CORNER_110, bv);
  gp1.set_corner_vertex(k0, k1-1, CORNER_010, bv);
  return bv;
}

// ----------------------------------------------------------------------------
//
template <class Data_Type>
Index CSurface<Data_Type>::add_cap_vertex_l2(Index bv,
					     Index k0, Index k1, Index k2,
					     Grid_Cell_List &gp1)
{
  if (bv == no_vertex)
    bv = create_vertex(k0,k1,k2);
  gp1.set_corner_vertex(k0-1, k1-1, CORNER_110, bv);
  gp1.set_corner_vertex(k0-1, k1, CORNER_100, bv);
  gp1.set_corner_vertex(k0, k1-1, CORNER_010, bv);
  gp1.set_corner_vertex(k0, k1, CORNER_000, bv);
  return bv;
}

// ----------------------------------------------------------------------------
//
template <class Data_Type>
Index CSurface<Data_Type>::add_cap_vertex_r2(Index bv,
					     Index k0, Index k1, Index k2,
					     Grid_Cell_List &gp0)
{
  if (bv == no_vertex)
    bv = create_vertex(k0,k1,k2);
  gp0.set_corner_vertex(k0-1, k1-1, CORNER_111, bv);
  gp0.set_corner_vertex(k0-1, k1, CORNER_101, bv);
  gp0.set_corner_vertex(k0, k1-1, CORNER_011, bv);
  gp0.set_corner_vertex(k0, k1, CORNER_001, bv);
  return bv;
}

// ----------------------------------------------------------------------------
//
template <class Data_Type>
void CSurface<Data_Type>::make_triangles(Grid_Cell_List &gp0, Index k2)
{
  Stride step0 = stride[0], step1 = stride[1], step2 = stride[2];
  std::vector<Grid_Cell *> &clist = gp0.cells;
  Index cc = gp0.cell_count;
  const Data_Type *g0 = grid + (k2-1)*step2;
  Stride step01 = step0 + step1;
  for (Index k = 0 ; k < cc ; ++k)
    {
      Grid_Cell *c = clist[k];
      const Data_Type *gc = g0 + step0*(Stride)c->k0 + step1*(Stride)c->k1, *gc2 = gc + step2;
      int bits = ((inside(gc[0]) ? 1 : 0) |
		  (inside(gc[step0]) ? 2 : 0) |
		  (inside(gc[step01]) ? 4 : 0) |
		  (inside(gc[step1]) ? 8 : 0) |
		  (inside(gc2[0]) ? 16 : 0) |
		  (inside(gc2[step0]) ? 32 : 0) |
		  (inside(gc2[step01]) ? 64 : 0) |
		  (inside(gc2[step1]) ? 128 : 0));

      Index *cell_vertices = c->vertex;
      int *t = triangle_table[bits];
      for (int e = *t ; e != -1 ; ++t, e = *t)
	add_triangle_corner(cell_vertices[e]);

      if (c->boundary && cap_faces)
	make_cap_triangles(c->k0, c->k1, k2, bits, cell_vertices);
    }
}

// ----------------------------------------------------------------------------
//
template <class Data_Type>
void CSurface<Data_Type>::make_triangles(Grid_Cell_List &gp0, Index k2, Region_Id region_id)
{
  Stride step0 = stride[0], step1 = stride[1], step2 = stride[2];
  std::vector<Grid_Cell *> &clist = gp0.cells;
  Index cc = gp0.cell_count;
  const Data_Type *g0 = grid + (k2-1)*step2;
  Stride step01 = step0 + step1;
  for (Index k = 0 ; k < cc ; ++k)
    {
      Grid_Cell *c = clist[k];
      const Data_Type *gc = g0 + step0*(Stride)c->k0 + step1*(Stride)c->k1, *gc2 = gc + step2;
      int bits = ((inside(gc[0]) == region_id ? 1 : 0) |
		  (inside(gc[step0]) == region_id ? 2 : 0) |
		  (inside(gc[step01]) == region_id ? 4 : 0) |
		  (inside(gc[step1]) == region_id ? 8 : 0) |
		  (inside(gc2[0]) == region_id ? 16 : 0) |
		  (inside(gc2[step0]) == region_id ? 32 : 0) |
		  (inside(gc2[step01]) == region_id ? 64 : 0) |
		  (inside(gc2[step1]) == region_id ? 128 : 0));

      Index *cell_vertices = c->vertex;
      int *t = triangle_table[bits];
      for (int e = *t ; e != -1 ; ++t, e = *t)
	add_triangle_corner(cell_vertices[e]);

      if (c->boundary && cap_faces)
	make_cap_triangles(c->k0, c->k1, k2, bits, cell_vertices);
    }
}

// ----------------------------------------------------------------------------
//
template <class Data_Type>
inline void CSurface<Data_Type>::make_cap_triangles(Index k0, Index k1, Index k2, int bits,
						    Index *cell_vertices)
{
  // Check 6 faces for being on boundary, assemble 4 bits for
  // face and call triangle building routine.
  Index k0_size = size[0], k1_size = size[1], k2_size = size[2];
  if (k0 == 0)
    make_cap_triangles(4, bits, cell_vertices);
  if (k0 + 2 == k0_size)
    make_cap_triangles(2, bits, cell_vertices);
  if (k1 == 0)
    make_cap_triangles(1, bits, cell_vertices);
  if (k1 + 2 == k1_size)
    make_cap_triangles(3, bits, cell_vertices);
  if (k2 == 1)
    make_cap_triangles(0, bits, cell_vertices);
  if (k2 + 1 == k2_size)
    make_cap_triangles(5, bits, cell_vertices);
}


// ----------------------------------------------------------------------------
//
template <class Data_Type>
void contour_surface(const Reference_Counted_Array::Array<Data_Type> &data,
		     int value, const IArray &surface_ids,
		     bool cap_faces, Contour_Surface **cs)
{
  // contouring calculation requires contiguous array
  // put sizes in x, y, z order
  Index size[3] = {static_cast<Index>(data.size(2)),
		   static_cast<Index>(data.size(1)),
		   static_cast<Index>(data.size(0))};
  Stride stride[3] = {data.stride(2), data.stride(1), data.stride(0)};
  int *surf_ids = (surface_ids.dimension() == 1 ? surface_ids.values() : NULL);
  Region_Test<Data_Type> inside((Data_Type)value, surf_ids);
  Contour_Surface *csurf = new CSurface<Data_Type>(data.values(), size, stride, inside, cap_faces);
  csurf->compute_surface();
  *cs = csurf;
}

// ----------------------------------------------------------------------------
//
template <class Data_Type>
void contour_surfaces(const Reference_Counted_Array::Array<Data_Type> &data,
		      const IArray &surface_ids, bool cap_faces, Contour_Surface **cs)
{
  // contouring calculation requires contiguous array
  // put sizes in x, y, z order
  Index size[3] = {static_cast<Index>(data.size(2)),
		   static_cast<Index>(data.size(1)),
		   static_cast<Index>(data.size(0))};
  Stride stride[3] = {data.stride(2), data.stride(1), data.stride(0)};
  int *surf_ids = (surface_ids.dimension() == 1 ? surface_ids.values() : NULL);
  Region_Test<Data_Type> inside(surf_ids);
  Contour_Surface *csurf = new CSurface<Data_Type>(data.values(), size, stride, inside, cap_faces);
  csurf->compute_surfaces();
  *cs = csurf;
}

// ----------------------------------------------------------------------------
//
static PyObject *python_surface(const Region_Surface &surf, bool include_id = false)
{
  float *vxyz;
  int *tvi;
  size_t nv = surf.vertices.size()/3, nt = surf.triangles.size()/3;
  PyObject *vertex_xyz = python_float_array(nv, 3, &vxyz);
  PyObject *tv_indices = python_int_array(nt, 3, &tvi);

  size_t nv3 = 3*nv, nt3 = 3*nt;
  for (size_t i = 0 ; i < nv3 ; ++i)
    vxyz[i] = surf.vertices[i];
  for (size_t i = 0 ; i < nt3 ; ++i)
    tvi[i] = surf.triangles[i];

  PyObject *py_surf = (include_id ?
		       python_tuple(PyLong_FromLong(surf.region_id), vertex_xyz, tv_indices) :
		       python_tuple(vertex_xyz, tv_indices));
  return py_surf;
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *
segmentation_surface(PyObject *, PyObject *args, PyObject *keywds)
{
  Numeric_Array region_map;
  int value;
  IArray groups;
  const char *kwlist[] = {"region_map", "index", "groups", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&p|O&"),
				   (char **)kwlist,
				   parse_3d_array, &region_map,
				   &value,
				   parse_int_n_array, &groups))
    return NULL;

  if (groups.dimension() == 1 && !groups.is_contiguous())
    {
      PyErr_Format(PyExc_ValueError, "segmentation_surface(): groups array argument must be contiguous");
      return NULL;
    }
  
  PyObject *surf;
  try
    {
      bool cap_faces = true;
  
      Contour_Surface *cs;
      Py_BEGIN_ALLOW_THREADS
	call_template_function(contour_surface, region_map.value_type(),
			       (region_map, value, groups, cap_faces, &cs));
      Py_END_ALLOW_THREADS

      surf = python_surface(cs->surfaces()[0]);

      Py_BEGIN_ALLOW_THREADS
      delete cs;
      Py_END_ALLOW_THREADS
    }
  catch (std::bad_alloc&)
    {
      PyErr_Format(PyExc_MemoryError,
		   "segmentation_surface(): Out of memory, region map size (%d,%d,%d)",
		   region_map.size(0), region_map.size(1), region_map.size(2));
      return NULL;
    }

  return surf;
}

// ----------------------------------------------------------------------------
//
extern "C" PyObject *
segmentation_surfaces(PyObject *, PyObject *args, PyObject *keywds)
{
  Numeric_Array region_map;
  IArray groups;
  const char *kwlist[] = {"region_map", "groups", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>("O&|O&"),
				   (char **)kwlist,
				   parse_3d_array, &region_map,
				   parse_int_n_array, &groups))
    return NULL;

  if (groups.dimension() == 1 && !groups.is_contiguous())
    {
      PyErr_Format(PyExc_ValueError, "segmentation_surfaces(): groups array argument must be contiguous");
      return NULL;
    }
  
  PyObject *surfs;
  try
    {
      bool cap_faces = true;
  
      Contour_Surface *cs;
      Py_BEGIN_ALLOW_THREADS
	call_template_function(contour_surfaces, region_map.value_type(),
			       (region_map, groups, cap_faces, &cs));
      Py_END_ALLOW_THREADS

      const Region_Surfaces &surfaces = cs->surfaces();
      size_t ns = surfaces.size();
      surfs = PyTuple_New(ns);
      for (size_t i = 0 ; i < ns ; ++i)
	PyTuple_SetItem(surfs, i, python_surface(surfaces[i], true));
      delete cs;
    }
  catch (std::bad_alloc&)
    {
      PyErr_Format(PyExc_MemoryError,
		   "segmentation_surfaces(): Out of memory, region map size (%d,%d,%d)",
		   region_map.size(0), region_map.size(1), region_map.size(2));
      return NULL;
    }

  return surfs;
}
