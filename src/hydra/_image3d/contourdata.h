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
