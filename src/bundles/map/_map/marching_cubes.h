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
// Calculate constant intensity surfaces for volumes.
// Uses the marching cubes algorithm.
//
#ifndef MARCHING_CUBES_HEADER_INCLUDED
#define MARCHING_CUBES_HEADER_INCLUDED

#include <math.h>		// use sqrt()

#include "contourdata.h"	// use cube_edges, triangle_table

namespace Contour_Calculation
{
  //#define CONTOUR_ARRAY_BLOCK_SIZE 65536
#define CONTOUR_ARRAY_BLOCK_SIZE 1048576

const VIndex no_vertex = ~(VIndex)0;

typedef unsigned int BIndex;  // Block_Array indexing.
  
// ----------------------------------------------------------------------------
// Use array broken into blocks for storing vertices and triangles.
// Might be better to use std::vector but Block_Array avoids reallocating
// the array as it grows.
//
template <class T> class Block_Array
{
public:
  Block_Array(BIndex block_size);
  ~Block_Array();
  BIndex size() { return ae + afsize; }
  T element(BIndex k)
    { return (k >= afsize ? a[k-afsize] : alist[k/bsize][k%bsize]); }
  void set_element(BIndex k, T e)
    { if (k >= afsize) a[k-afsize] = e;
      else alist[k/bsize][k%bsize] = e; }
  void add_element(T e)
    {
      if (ae == bsize || alsize == 0) next_block();
      a[ae++] = e;
    }
  void array(T *carray);  // Contiguous array.
  void reset() // Does not deallocate memory.
    { ae = anxt = afsize = 0; a = (alist ? alist[0] : a); }
private:
  BIndex bsize; // block size in elements.
  BIndex ae;    // elements in last block in use.
  BIndex anxt;  // next unused block number.
  BIndex ale;   // number of blocks allocated.
  BIndex alsize; // size of array of block pointers.
  BIndex afsize; // number of elements in full blocks.
  T *a;		// last block in use.
  T **alist;	// pointers to allocated blocks.
  void next_block();
};

// ----------------------------------------------------------------------------
//
template <class T> Block_Array<T>::Block_Array(BIndex block_size)
{
  this->bsize = block_size;
  this->ae = this->anxt = this->ale = this->alsize = this->afsize = 0;
  this->a = NULL;
  this->alist = NULL;
}

// ----------------------------------------------------------------------------
//
  template <class T> Block_Array<T>::~Block_Array()
{
  for (BIndex k = 0 ; k < ale ; ++k)
    delete [] alist[k];
  delete [] alist;
}

// ----------------------------------------------------------------------------
//
template <class T> void Block_Array<T>::next_block()
{
  if (anxt >= ale)
    {
      if (alsize == 0)
	{
	  this->alsize = 1024;
	  this->alist = new T*[alsize];
	}
      if (ale == alsize)
	{
	  T **alist2 = new T*[2*alsize];
	  for (BIndex k = 0 ; k < alsize ; ++k)
	    alist2[k] = alist[k];
	  delete [] alist;
	  this->alist = alist2;
	  this->alsize *= 2;
	}
      alist[ale++] = new T[bsize];
    }
  this->a = alist[anxt++];
  this->ae = 0;
  this->afsize = (anxt - 1) * bsize;
}

// ----------------------------------------------------------------------------
//
template <class T> void Block_Array<T>::array(T *carray)
{
  BIndex k = 0;
  for (BIndex i = 0 ; i+1 < anxt ; ++i)
	{
	  T *b = alist[i];
	  for (BIndex j = 0 ; j < bsize ; ++j, ++k)
	    carray[k] = b[j];
	}
  for (BIndex j = 0 ; j < ae ; ++j, ++k)
    carray[k] = a[j];
}

// ----------------------------------------------------------------------------
// A cell is a cube in the 3D data array with corners at 8 grid points.
// Grid_Cell records the vertex numbers on cube edges and at corners needed
// for triangulating the surface within the cell including triangulating
// boundary faces of the 3D array.
//
class Grid_Cell
{
public:
  AIndex k0, k1;	// Cell position in xy plane.
  VIndex vertex[20];	// Vertex numbers for 12 edges and 8 corners.
  bool boundary;	// Contour reaches boundary.
};

typedef int64_t CIndex;  // Index into 2D array of Grid_Cell
  
// ----------------------------------------------------------------------------
// 2D array of grid cells.  Each grid cell records the vertex numbers along
// the cube edges and corners needed for triangulating the surface within the cell.
//
class Grid_Cell_List
{
public:
  Grid_Cell_List(AIndex size0, AIndex size1) : cells(CONTOUR_ARRAY_BLOCK_SIZE)
  {
    this->cell_table_size0 = size0+2;	// Pad by one grid cell.
    AIndex cell_table_size1 = size1+2;
    GIndex size = cell_table_size0 * cell_table_size1;
    this->cell_count = 0;
    this->cell_base_index = 2;
    this->cell_table = new CIndex[size];
    for (GIndex i = 0 ; i < size ; ++i)
      cell_table[i] = no_cell;
    for (GIndex i = 0 ; i < cell_table_size0 ; ++i)
      cell_table[i] = cell_table[size-i-1] = out_of_bounds;
    for (GIndex i = 0 ; i < size ; i += cell_table_size0)
      cell_table[i] = cell_table[i+cell_table_size0-1] = out_of_bounds;
  }
  ~Grid_Cell_List()
    {
      delete_cells();
      delete [] cell_table;
    }
  void set_edge_vertex(AIndex k0, AIndex k1, Edge_Number e, VIndex v)
  {
    Grid_Cell *c = cell(k0,k1);
    if (c)
      c->vertex[e] = v;
  }
  void set_corner_vertex(AIndex k0, AIndex k1, Corner_Number corner, VIndex v)
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

  GIndex cell_count;		// Number of elements of cells currently in use.
  Block_Array<Grid_Cell *> cells;

private:
  static const CIndex out_of_bounds = 0;
  static const CIndex no_cell = 1;
  AIndex cell_table_size0;
  CIndex cell_base_index;	// Minimum valid cell index.
  CIndex *cell_table;		// Maps cell plane index to cell list index.

  // Get cell, initializing or allocating a new one if necessary.
  Grid_Cell *cell(AIndex k0, AIndex k1)
  {
    CIndex i = k0+1 + (k1+1)*cell_table_size0;
    CIndex c = cell_table[i];
    if (c == out_of_bounds)
      return NULL;

    Grid_Cell *cp;
    if (c != no_cell && c >= cell_base_index)
      cp = cells.element(c-cell_base_index);
    else
      {
	cell_table[i] = cell_base_index + cell_count;
	if (cell_count < cells.size())
	  cp = cells.element(cell_count);
	else
	  cells.add_element(cp = new Grid_Cell);
	cp->k0 = k0; cp->k1 = k1; cp->boundary = false;
	cell_count += 1;
      }
    return cp;
  }

  void delete_cells()
  {
    CIndex cc = cells.size();
    for (CIndex c = 0 ; c < cc ; ++c)
      delete cells.element(c);
  }
};

// ----------------------------------------------------------------------------
//
template <class Data_Type>
class CSurface : public Contour_Surface
{
public:
  CSurface(const Data_Type *grid, const AIndex size[3], const GIndex stride[3],
	   float threshold, bool cap_faces, BIndex block_size)
    : grid(grid), threshold(threshold), cap_faces(cap_faces),
      vxyz(3*block_size), tvi(3*block_size)
    {
      for (int a = 0 ; a < 3 ; ++a)
	{ this->size[a] = size[a]; this->stride[a] = stride[a]; }
      compute_surface();
    }
  virtual ~CSurface() {}

  virtual VIndex vertex_count() { return vxyz.size()/3; }
  virtual TIndex triangle_count() { return tvi.size()/3; }
  virtual void geometry(float *vertex_xyz, VIndex *triangle_vertex_indices)
    { vxyz.array(vertex_xyz); tvi.array(triangle_vertex_indices); }
  virtual void normals(float *normals);

private:
  const Data_Type *grid;
  AIndex size[3];
  GIndex stride[3];
  float threshold;
  bool cap_faces;
  Block_Array<float> vxyz;
  Block_Array<VIndex> tvi;

  void compute_surface();
  void mark_plane_edge_cuts(Grid_Cell_List &gp0, Grid_Cell_List &gp1, AIndex k2);
  void mark_interior_edge_cuts(AIndex k1, AIndex k2,
			       Grid_Cell_List &gp0, Grid_Cell_List &gp1);
  void mark_boundary_edge_cuts(AIndex k0, AIndex k1, AIndex k2,
			       Grid_Cell_List &gp0, Grid_Cell_List &gp1);

  void add_vertex_axis_0(AIndex k0, AIndex k1, AIndex k2, float x0,
			 Grid_Cell_List &gp0, Grid_Cell_List &gp1);
  void add_vertex_axis_1(AIndex k0, AIndex k1, AIndex k2, float x1,
			 Grid_Cell_List &gp0, Grid_Cell_List &gp1);
  void add_vertex_axis_2(AIndex k0, AIndex k1, float x2,
			 Grid_Cell_List &gp);

  VIndex add_cap_vertex_l0(VIndex bv, AIndex k0, AIndex k1, AIndex k2,
			  Grid_Cell_List &gp0, Grid_Cell_List &gp1);
  VIndex add_cap_vertex_r0(VIndex bv, AIndex k0, AIndex k1, AIndex k2,
			  Grid_Cell_List &gp0, Grid_Cell_List &gp1);
  VIndex add_cap_vertex_l1(VIndex bv, AIndex k0, AIndex k1, AIndex k2,
			  Grid_Cell_List &gp0, Grid_Cell_List &gp1);
  VIndex add_cap_vertex_r1(VIndex bv, AIndex k0, AIndex k1, AIndex k2,
			  Grid_Cell_List &gp0, Grid_Cell_List &gp1);
  VIndex add_cap_vertex_l2(VIndex bv, AIndex k0, AIndex k1, AIndex k2,
			  Grid_Cell_List &gp1);
  VIndex add_cap_vertex_r2(VIndex bv, AIndex k0, AIndex k1, AIndex k2,
			  Grid_Cell_List &gp0);

  void make_triangles(Grid_Cell_List &gp0, AIndex k2);
  void add_triangle_corner(VIndex v) { tvi.add_element(v); }
  VIndex create_vertex(float x, float y, float z)
    { vxyz.add_element(x); vxyz.add_element(y); vxyz.add_element(z);
      return vertex_count()-1; }
  void make_cap_triangles(int face, int bits, VIndex *cell_vertices)
    {
      int fbits = face_corner_bits[face][bits];
      int *t = cap_triangle_table[face][fbits];
      for (int v = *t ; v != -1 ; ++t, v = *t)
	add_triangle_corner(cell_vertices[v]);
    }
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
  // If grid point value is above threshold check if 6 connected edges
  // cross contour surface and make vertex, add vertex to 4 bordering
  // grid cells, triangulate grid cells between two z grid planes.
  //
  Grid_Cell_List gcp0(size[0]-1, size[1]-1), gcp1(size[0]-1, size[1]-1);
  for (AIndex k2 = 0 ; k2 < size[2] ; ++k2)
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
void CSurface<Data_Type>::mark_plane_edge_cuts(Grid_Cell_List &gp0,
					       Grid_Cell_List &gp1,
					       AIndex k2)
{
  AIndex k0_size = size[0], k1_size = size[1], k2_size = size[2];

  for (AIndex k1 = 0 ; k1 < k1_size ; ++k1)
    {
      if (k1 == 0 || k1+1 == k1_size || k2 == 0 || k2+1 == k2_size)
	for (AIndex k0 = 0 ; k0 < k0_size ; ++k0)
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
inline void CSurface<Data_Type>::mark_interior_edge_cuts(AIndex k1, AIndex k2,
							 Grid_Cell_List &gp0,
							 Grid_Cell_List &gp1)
{
  GIndex step0 = stride[0], step1 = stride[1], step2 = stride[2];
  AIndex k0_max = (size[0] > 0 ? size[0]-1 : 0);

  const Data_Type *g = grid + step2*(GIndex)k2 + step1*(GIndex)k1 + step0;
  for (AIndex k0 = 1 ; k0 < k0_max ; ++k0, g += step0)
    {
      float v0 = *g - threshold;
      if (!(v0 < 0))
	{
	  // Grid point value is above threshold.
	  // Look at 6 neigbors along x,y,z axes for values below threshold.
	  float v1;
	  if ((v1 = (float)(*(g-step0)-threshold)) < 0)
	    add_vertex_axis_0(k0-1, k1, k2, k0-v0/(v0-v1), gp0, gp1);
	  if ((v1 = (float)(g[step0]-threshold)) < 0)
	    add_vertex_axis_0(k0, k1, k2, k0+v0/(v0-v1), gp0, gp1);
	  if ((v1 = (float)(*(g-step1)-threshold)) < 0)
	    add_vertex_axis_1(k0, k1-1, k2, k1-v0/(v0-v1), gp0, gp1);
	  if ((v1 = (float)(g[step1]-threshold)) < 0)
	    add_vertex_axis_1(k0, k1, k2, k1+v0/(v0-v1), gp0, gp1);
	  if ((v1 = (float)(*(g-step2)-threshold)) < 0)
	    add_vertex_axis_2(k0, k1, k2-v0/(v0-v1), gp0);
	  if ((v1 = (float)(g[step2]-threshold)) < 0)
	    add_vertex_axis_2(k0, k1, k2+v0/(v0-v1), gp1);
	}
    }
}

// ----------------------------------------------------------------------------
// Compute edge cut vertices in 6 directions and capping corner vertex for
// boundary grid points.
//
template <class Data_Type>
inline void CSurface<Data_Type>::mark_boundary_edge_cuts(AIndex k0, AIndex k1, AIndex k2,
							 Grid_Cell_List &gp0,
							 Grid_Cell_List &gp1)
{
  GIndex step0 = stride[0], step1 = stride[1], step2 = stride[2];
  AIndex k0_size = size[0], k1_size = size[1], k2_size = size[2];
  const Data_Type *g = grid + step2*(GIndex)k2 + step1*(GIndex)k1 + step0*(GIndex)k0;
  float v0 = *g - threshold;
  if (v0 < 0)
    return;

  // Check 6 neighbor vertices for edge crossings.

  VIndex bv = no_vertex;
  float v1;

  // Axis 0 left
  if (k0 > 0)
    {
      if ((v1 = (float)(*(g-step0)-threshold)) < 0)
	add_vertex_axis_0(k0-1, k1, k2, k0-v0/(v0-v1), gp0, gp1);
    }
  else if (cap_faces)  // boundary vertex for capping box faces.
    bv = add_cap_vertex_l0(bv, k0, k1, k2, gp0, gp1);

  // Axis 0 right
  if (k0+1 < k0_size)
    {
      if ((v1 = (float)(g[step0]-threshold)) < 0)
	add_vertex_axis_0(k0, k1, k2, k0+v0/(v0-v1), gp0, gp1);
    }
  else if (cap_faces)
    bv = add_cap_vertex_r0(bv, k0, k1, k2, gp0, gp1);

  // Axis 1 left
  if (k1 > 0)
    {
      if ((v1 = (float)(*(g-step1)-threshold)) < 0)
	add_vertex_axis_1(k0, k1-1, k2, k1-v0/(v0-v1), gp0, gp1);
    }
  else if (cap_faces)
    bv = add_cap_vertex_l1(bv, k0, k1, k2, gp0, gp1);

  // Axis 1 right
  if (k1+1 < k1_size)
    {
      if ((v1 = (float)(g[step1]-threshold)) < 0)
	add_vertex_axis_1(k0, k1, k2, k1+v0/(v0-v1), gp0, gp1);
    }
  else if (cap_faces)
    bv = add_cap_vertex_r1(bv, k0, k1, k2, gp0, gp1);

  // Axis 2 left
  if (k2 > 0)
    {
      if ((v1 = (float)(*(g-step2)-threshold)) < 0)
	add_vertex_axis_2(k0, k1, k2-v0/(v0-v1), gp0);
    }
  else if (cap_faces)
    bv = add_cap_vertex_l2(bv, k0, k1, k2, gp1);

  // Axis 2 right
  if (k2+1 < k2_size)
    {
      if ((v1 = (float)(g[step2]-threshold)) < 0)
	add_vertex_axis_2(k0, k1, k2+v0/(v0-v1), gp1);
    }
  else if (cap_faces)
    bv = add_cap_vertex_r2(bv, k0, k1, k2, gp0);
}

// ----------------------------------------------------------------------------
// Add axis 0 edge cut to four adjoining grid cells.
//
template <class Data_Type>
void CSurface<Data_Type>::add_vertex_axis_0(AIndex k0, AIndex k1, AIndex k2, float x0,
					    Grid_Cell_List &gp0, Grid_Cell_List &gp1)
{
  VIndex v = create_vertex(x0,k1,k2);
  gp0.set_edge_vertex(k0, k1-1, EDGE_A11, v);
  gp0.set_edge_vertex(k0, k1, EDGE_A01, v);
  gp1.set_edge_vertex(k0, k1-1, EDGE_A10, v);
  gp1.set_edge_vertex(k0, k1, EDGE_A00, v);
}

// ----------------------------------------------------------------------------
// Add axis 1 edge cut to four adjoining grid cells.
//
template <class Data_Type>
void CSurface<Data_Type>::add_vertex_axis_1(AIndex k0, AIndex k1, AIndex k2, float x1,
					    Grid_Cell_List &gp0, Grid_Cell_List &gp1)
{
  VIndex v = create_vertex(k0,x1,k2);
  gp0.set_edge_vertex(k0-1, k1, EDGE_1A1, v);
  gp0.set_edge_vertex(k0, k1, EDGE_0A1, v);
  gp1.set_edge_vertex(k0-1, k1, EDGE_1A0, v);
  gp1.set_edge_vertex(k0, k1, EDGE_0A0, v);
}

// ----------------------------------------------------------------------------
// Add axis 2 edge cut to four adjoining grid cells.
//
template <class Data_Type>
void CSurface<Data_Type>::add_vertex_axis_2(AIndex k0, AIndex k1, float x2,
					    Grid_Cell_List &gp)
{
  VIndex v = create_vertex(k0,k1,x2);
  gp.set_edge_vertex(k0, k1, EDGE_00A, v);
  gp.set_edge_vertex(k0-1, k1, EDGE_10A, v);
  gp.set_edge_vertex(k0, k1-1, EDGE_01A, v);
  gp.set_edge_vertex(k0-1, k1-1, EDGE_11A, v);
}

// ----------------------------------------------------------------------------
//
template <class Data_Type>
VIndex CSurface<Data_Type>::add_cap_vertex_l0(VIndex bv,
					     AIndex k0, AIndex k1, AIndex k2,
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
VIndex CSurface<Data_Type>::add_cap_vertex_r0(VIndex bv,
					     AIndex k0, AIndex k1, AIndex k2,
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
VIndex CSurface<Data_Type>::add_cap_vertex_l1(VIndex bv,
					     AIndex k0, AIndex k1, AIndex k2,
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
VIndex CSurface<Data_Type>::add_cap_vertex_r1(VIndex bv,
					     AIndex k0, AIndex k1, AIndex k2,
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
VIndex CSurface<Data_Type>::add_cap_vertex_l2(VIndex bv,
					     AIndex k0, AIndex k1, AIndex k2,
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
VIndex CSurface<Data_Type>::add_cap_vertex_r2(VIndex bv,
					     AIndex k0, AIndex k1, AIndex k2,
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
void CSurface<Data_Type>::make_triangles(Grid_Cell_List &gp0, AIndex k2)
{
  GIndex step0 = stride[0], step1 = stride[1], step2 = stride[2];
  AIndex k0_size = size[0], k1_size = size[1], k2_size = size[2];
  Block_Array<Grid_Cell *> &clist = gp0.cells;
  CIndex cc = gp0.cell_count;
  const Data_Type *g0 = grid + step2*(GIndex)(k2-1);
  GIndex step01 = step0 + step1;
  for (CIndex k = 0 ; k < cc ; ++k)
    {
      Grid_Cell *c = clist.element(k);
      const Data_Type *gc = g0 + step0*(GIndex)c->k0 + step1*(GIndex)c->k1, *gc2 = gc + step2;
      int bits = ((gc[0] < threshold ? 0 : 1) |
		  (gc[step0] < threshold ? 0 : 2) |
		  (gc[step01] < threshold ? 0 : 4) |
		  (gc[step1] < threshold ? 0 : 8) |
		  (gc2[0] < threshold ? 0 : 16) |
		  (gc2[step0] < threshold ? 0 : 32) |
		  (gc2[step01] < threshold ? 0 : 64) |
		  (gc2[step1] < threshold ? 0 : 128));

      VIndex *cell_vertices = c->vertex;
      int *t = triangle_table[bits];
      for (int e = *t ; e != -1 ; ++t, e = *t)
	add_triangle_corner(cell_vertices[e]);

      if (c->boundary && cap_faces)
	{
	  // Check 6 faces for being on boundary, assemble 4 bits for
	  // face and call triangle building routine.
	  if (c->k0 == 0)
	    make_cap_triangles(4, bits, cell_vertices);
	  if (c->k0 + 2 == k0_size)
	    make_cap_triangles(2, bits, cell_vertices);
	  if (c->k1 == 0)
	    make_cap_triangles(1, bits, cell_vertices);
	  if (c->k1 + 2 == k1_size)
	    make_cap_triangles(3, bits, cell_vertices);
	  if (k2 == 1)
	    make_cap_triangles(0, bits, cell_vertices);
	  if (k2 + 1 == k2_size)
	    make_cap_triangles(5, bits, cell_vertices);
	}
    }
}

// ----------------------------------------------------------------------------
// Normals are negative of symmetric difference data gradient.
//
template <class Data_Type>
void CSurface<Data_Type>::normals(float *normals)
{
  int64_t n3 = 3*vertex_count();
  for (int64_t v = 0 ; v < n3 ; v += 3)
    {  
      float x[3] = {vxyz.element(v), vxyz.element(v+1), vxyz.element(v+2)};
      float g[3];
      for (int a = 0 ; a < 3 ; ++a)
	g[a] = (x[a] == 0 ? 1 : (x[a] == size[a]-1 ? -1 : 0));
      if (g[0] == 0 && g[1] == 0 && g[2] == 0)
	{
	  AIndex i[3] = {(AIndex)x[0], (AIndex)x[1], (AIndex)x[2]};
	  const Data_Type *ga = grid + stride[0]*(GIndex)i[0]+stride[1]*(GIndex)i[1]+stride[2]*(GIndex)i[2];
	  const Data_Type *gb = ga;
	  AIndex off[3] = {0,0,0};
	  float fb = 0;
	  for (int a = 0 ; a < 3 ; ++a)
	    if ((fb = x[a]-i[a]) > 0) { off[a] = 1; gb = ga + stride[a]; break; }
	  float fa = 1-fb;
	  for (int a = 0 ; a < 3 ; ++a)
	    {
	      GIndex s = stride[a];
	      AIndex ia = i[a], ib = ia + off[a];
	      g[a] = (fa*(ia == 0 ?
			  2*((float)ga[s]-ga[0]) : (float)ga[s]-*(ga-s))
		      + fb*(ib == 0 ? 2*((float)gb[s]-gb[0]) :
			    ib == size[a]-1 ? 2*((float)gb[0]-*(gb-s))
			    : (float)gb[s]-*(gb-s)));
	    }
	  float norm = sqrt(g[0]*g[0] + g[1]*g[1] + g[2]*g[2]);
	  if (norm > 0)
	    { g[0] /= norm; g[1] /= norm; g[2] /= norm;}
	}
      normals[v] = -g[0]; normals[v+1] = -g[1]; normals[v+2] = -g[2];
    }
}

} // end of namespace Contour_Calculation

#endif // MARCHING_CUBES_HEADER_INCLUDED
