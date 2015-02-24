// vi: set expandtab shiftwidth=4 softtabstop=4:
// ----------------------------------------------------------------------------
// Calculate constant intensity surfaces for volumes.
// Uses the marching cubes algorithm.
//
//#include <iostream>		// use std::cerr for debugging

#include <math.h>		// use sqrt()

#include "contourdata.h"	// use cube_edges, triangle_table

namespace Contour_Calculation
{
  //#define CONTOUR_ARRAY_BLOCK_SIZE 65536
#define CONTOUR_ARRAY_BLOCK_SIZE 1048576

const Index no_vertex = ~(Index)0;

// ----------------------------------------------------------------------------
// Use array broken into blocks for storing vertices and triangles.
//
template <class T> class Block_Array
{
public:
  Block_Array(Index block_size);
  ~Block_Array();
  Index size() { return ae + afsize; }
  T element(Index k)
    { return (k >= afsize ? a[k-afsize] : alist[k/bsize][k%bsize]); }
  void set_element(Index k, T e)
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
  Index bsize; // block size in elements.
  Index ae;    // elements in last block in use.
  Index anxt;  // next unused block number.
  Index ale;   // number of blocks allocated.
  Index alsize; // size of array of block pointers.
  Index afsize; // number of elements in full blocks.
  T *a;		// last block in use.
  T **alist;	// pointers to allocated blocks.
  void next_block();
};

// ----------------------------------------------------------------------------
//
template <class T> Block_Array<T>::Block_Array(Index block_size)
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
  for (Index k = 0 ; k < ale ; ++k)
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
	  for (Index k = 0 ; k < alsize ; ++k)
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
  Index k = 0;
  for (Index i = 0 ; i+1 < anxt ; ++i)
	{
	  T *b = alist[i];
	  for (Index j = 0 ; j < bsize ; ++j, ++k)
	    carray[k] = b[j];
	}
  for (Index j = 0 ; j < ae ; ++j, ++k)
    carray[k] = a[j];
}

// ----------------------------------------------------------------------------
//
class Grid_Cell
{
public:
  Index k0, k1;		// Cell position in xy plane.
  Index vertex[20];	// Vertex numbers for 12 edges and 8 corners.
  bool boundary;	// Contour reaches boundary.
};

// ----------------------------------------------------------------------------
//
class Grid_Cell_List
{
public:
  Grid_Cell_List(Index size0, Index size1) : cells(CONTOUR_ARRAY_BLOCK_SIZE)
  {
    this->rsize = size0+2;	// Pad by one grid cell.
    Index csize = size1+2;
    Index size = rsize * csize;
    this->cell_count = 0;
    this->cmin = 2;
    this->cell_table = new Index[size];
    for (Index i = 0 ; i < size ; ++i)
      cell_table[i] = no_cell;
    for (Index i = 0 ; i < rsize ; ++i)
      cell_table[i] = cell_table[size-i-1] = out_of_bounds;
    for (Index i = 0 ; i < size ; i += rsize)
      cell_table[i] = cell_table[i+rsize-1] = out_of_bounds;
  }
  ~Grid_Cell_List()
    {
      delete_cells();
      delete [] cell_table;
    }
  void set_edge_vertex(Index k0, Index k1, Index e, Index v)
  {
    Grid_Cell *c = cell(k0,k1);
    if (c)
      c->vertex[e] = v;
  }
  Grid_Cell *cell(Index k0, Index k1)
  {
    Index i = k0+1 + (k1+1)*rsize;
    Index c = cell_table[i];
    if (c == out_of_bounds)
      return NULL;

    Grid_Cell *cp;
    if (c != no_cell && c >= cmin)
      cp = cells.element(c-cmin);
    else
      {
	cell_table[i] = cmin + cell_count;
	if (cell_count < cells.size())
	  cp = cells.element(cell_count);
	else
	  cells.add_element(cp = new Grid_Cell);
	cp->k0 = k0; cp->k1 = k1; cp->boundary = false;
	cell_count += 1;
      }
    return cp;
  }
  void set_corner_vertex(Index k0, Index k1, Index corner, Index v)
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
      cmin += cell_count;
      cell_count = 0;
    }

  Index cell_count;
  Block_Array<Grid_Cell *> cells;

private:
  static const Index out_of_bounds = 0;
  static const Index no_cell = 1;
  Index rsize, cmin;
  Index *cell_table; // Maps cell plane index to cell list index.

  void delete_cells()
  {
    Index cc = cells.size();
    for (Index c = 0 ; c < cc ; ++c)
      delete cells.element(c);
  }
};

// ----------------------------------------------------------------------------
//
template <class Data_Type>
class CSurface : public Contour_Surface
{
public:
  CSurface(const Data_Type *grid, const Index size[3], const Stride stride[3],
	   float threshold, bool cap_faces, Index block_size)
    : grid(grid), threshold(threshold), cap_faces(cap_faces),
      vxyz(3*block_size), tvi(3*block_size)
    {
      for (int a = 0 ; a < 3 ; ++a)
	{ this->size[a] = size[a]; this->stride[a] = stride[a]; }
      compute_surface();
    }
  virtual ~CSurface() {}

  virtual Index vertex_count() { return vxyz.size()/3; }
  virtual Index triangle_count() { return tvi.size()/3; }
  virtual void geometry(float *vertex_xyz, Index *triangle_vertex_indices)
    { vxyz.array(vertex_xyz); tvi.array(triangle_vertex_indices); }
  virtual void normals(float *normals);

private:
  const Data_Type *grid;
  Index size[3];
  Stride stride[3];
  float threshold;
  bool cap_faces;
  Block_Array<float> vxyz;
  Block_Array<Index> tvi;

  void compute_surface();
  void mark_plane_edge_cuts(Grid_Cell_List &gp0, Grid_Cell_List &gp1, Index k2);
  void mark_edge_cuts(Index k0, Index k1, Index k2,
		      Grid_Cell_List &gp0, Grid_Cell_List &gp1);
  void add_vertex_0(Index k0, Index k1, Index k2, float x0,
		    Grid_Cell_List &gp0, Grid_Cell_List &gp1);
  void add_vertex_1(Index k0, Index k1, Index k2, float x1,
		    Grid_Cell_List &gp0, Grid_Cell_List &gp1);
  void add_vertex_2(Index k0, Index k1, float x2, Grid_Cell_List &gp);
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
  void add_triangle_corner(Index v) { tvi.add_element(v); }
  Index create_vertex(float x, float y, float z)
    { vxyz.add_element(x); vxyz.add_element(y); vxyz.add_element(z);
      return vertex_count()-1; }
  void make_cap_triangles(int face, int bits, Index *cell_vertices)
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
void CSurface<Data_Type>::mark_plane_edge_cuts(Grid_Cell_List &gp0,
					       Grid_Cell_List &gp1,
					       Index k2)
{
  Stride step0 = stride[0], step1 = stride[1], step2 = stride[2];
  Index k0_size = size[0], k1_size = size[1], k2_size = size[2];

  for (Index k1 = 0 ; k1 < k1_size ; ++k1)
    {
      if (k1 == 0 || k1+1 == k1_size || k2 == 0 || k2+1 == k2_size)
	for (Index k0 = 0 ; k0 < k0_size ; ++k0)
	  mark_edge_cuts(k0, k1, k2, gp0, gp1);
      else
	{
	  if (k0_size > 0)
	    mark_edge_cuts(0, k1, k2, gp0, gp1);
	  // Check interior edge crossing.
	  const Data_Type *g = grid + step2*k2 + step1*k1 + step0;
	  for (Index k0 = 1 ; k0+1 < k0_size ; ++k0, g += step0)
	    {
	      float v0 = *g - threshold;
	      if (!(v0 < 0))
		{
		  float v1;
		  if ((v1 = (float)(*(g-step0)-threshold)) < 0)
		    add_vertex_0(k0-1, k1, k2, k0-v0/(v0-v1), gp0, gp1);
		  if ((v1 = (float)(g[step0]-threshold)) < 0)
		    add_vertex_0(k0, k1, k2, k0+v0/(v0-v1), gp0, gp1);
		  if ((v1 = (float)(*(g-step1)-threshold)) < 0)
		    add_vertex_1(k0, k1-1, k2, k1-v0/(v0-v1), gp0, gp1);
		  if ((v1 = (float)(g[step1]-threshold)) < 0)
		    add_vertex_1(k0, k1, k2, k1+v0/(v0-v1), gp0, gp1);
		  if ((v1 = (float)(*(g-step2)-threshold)) < 0)
		    add_vertex_2(k0, k1, k2-v0/(v0-v1), gp0);
		  if ((v1 = (float)(g[step2]-threshold)) < 0)
		    add_vertex_2(k0, k1, k2+v0/(v0-v1), gp1);
		}
	    }
	  if (k0_size > 1)
	    mark_edge_cuts(k0_size-1, k1, k2, gp0, gp1);
	}
    }
}

// ----------------------------------------------------------------------------
//
template <class Data_Type>
void CSurface<Data_Type>::mark_edge_cuts(Index k0, Index k1, Index k2,
					 Grid_Cell_List &gp0,
					 Grid_Cell_List &gp1)
{
  Stride step0 = stride[0], step1 = stride[1], step2 = stride[2];
  Index k0_size = size[0], k1_size = size[1], k2_size = size[2];
  const Data_Type *g = grid + step2*k2 + step1*k1 + step0*k0;
  float v0 = *g - threshold;
  if (!(v0 < 0))
	    {
	      // Check 6 neighbor vertices for edge crossings.

	      Index bv = no_vertex;
	      float v1;
	      // TODO: Removing the k0,k1 bounds checks in the following
	      // six conditionals increased speed 40% on Intel Mac 10.4.10
	      // (benchmark score 175 went up to 196).
	      if (k0 > 0)
		{
		  if ((v1 = (float)(*(g-step0)-threshold)) < 0)
		    add_vertex_0(k0-1, k1, k2, k0-v0/(v0-v1), gp0, gp1);
		}
	      else if (cap_faces)  // boundary vertex for capping box faces.
		bv = add_cap_vertex_l0(bv, k0, k1, k2, gp0, gp1);
	      if (k0+1 < k0_size)
		{
		  if ((v1 = (float)(g[step0]-threshold)) < 0)
		    add_vertex_0(k0, k1, k2, k0+v0/(v0-v1), gp0, gp1);
		}
	      else if (cap_faces)
		bv = add_cap_vertex_r0(bv, k0, k1, k2, gp0, gp1);

	      if (k1 > 0)
		{
		  if ((v1 = (float)(*(g-step1)-threshold)) < 0)
		    add_vertex_1(k0, k1-1, k2, k1-v0/(v0-v1), gp0, gp1);
		}
	      else if (cap_faces)
		bv = add_cap_vertex_l1(bv, k0, k1, k2, gp0, gp1);
	      if (k1+1 < k1_size)
		{
		  if ((v1 = (float)(g[step1]-threshold)) < 0)
		    add_vertex_1(k0, k1, k2, k1+v0/(v0-v1), gp0, gp1);
		}
	      else if (cap_faces)
		bv = add_cap_vertex_r1(bv, k0, k1, k2, gp0, gp1);

	      if (k2 > 0)
		{
		  if ((v1 = (float)(*(g-step2)-threshold)) < 0)
		    add_vertex_2(k0, k1, k2-v0/(v0-v1), gp0);
		}
	      else if (cap_faces)
		bv = add_cap_vertex_l2(bv, k0, k1, k2, gp1);
	      if (k2+1 < k2_size)
		{
		  if ((v1 = (float)(g[step2]-threshold)) < 0)
		    add_vertex_2(k0, k1, k2+v0/(v0-v1), gp1);
		}
	      else if (cap_faces)
		bv = add_cap_vertex_r2(bv, k0, k1, k2, gp0);
	    }
}

// ----------------------------------------------------------------------------
//
template <class Data_Type>
void CSurface<Data_Type>::add_vertex_0(Index k0, Index k1, Index k2, float x0,
				       Grid_Cell_List &gp0, Grid_Cell_List &gp1)
{
  Index v = create_vertex(x0,k1,k2);
  gp0.set_edge_vertex(k0, k1-1, 6, v);
  gp0.set_edge_vertex(k0, k1, 4, v);
  gp1.set_edge_vertex(k0, k1-1, 2, v);
  gp1.set_edge_vertex(k0, k1, 0, v);
}

// ----------------------------------------------------------------------------
//
template <class Data_Type>
void CSurface<Data_Type>::add_vertex_1(Index k0, Index k1, Index k2, float x1,
				       Grid_Cell_List &gp0, Grid_Cell_List &gp1)
{
  Index v = create_vertex(k0,x1,k2);
  gp0.set_edge_vertex(k0-1, k1, 5, v);
  gp0.set_edge_vertex(k0, k1, 7, v);
  gp1.set_edge_vertex(k0-1, k1, 1, v);
  gp1.set_edge_vertex(k0, k1, 3, v);
}

// ----------------------------------------------------------------------------
//
template <class Data_Type>
void CSurface<Data_Type>::add_vertex_2(Index k0, Index k1, float x2,
				       Grid_Cell_List &gp)
{
  Index v = create_vertex(k0,k1,x2);
  gp.set_edge_vertex(k0, k1, 8, v);
  gp.set_edge_vertex(k0-1, k1, 9, v);
  gp.set_edge_vertex(k0, k1-1, 11, v);
  gp.set_edge_vertex(k0-1, k1-1, 10, v);
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
  gp0.set_corner_vertex(k0, k1-1, 7, bv);
  gp0.set_corner_vertex(k0, k1, 4, bv);
  gp1.set_corner_vertex(k0, k1-1, 3, bv);
  gp1.set_corner_vertex(k0, k1, 0, bv);
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
  gp0.set_corner_vertex(k0-1, k1-1, 6, bv);
  gp0.set_corner_vertex(k0-1, k1, 5, bv);
  gp1.set_corner_vertex(k0-1, k1-1, 2, bv);
  gp1.set_corner_vertex(k0-1, k1, 1, bv);
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
  gp0.set_corner_vertex(k0-1, k1, 5, bv);
  gp0.set_corner_vertex(k0, k1, 4, bv);
  gp1.set_corner_vertex(k0-1, k1, 1, bv);
  gp1.set_corner_vertex(k0, k1, 0, bv);
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
  gp0.set_corner_vertex(k0-1, k1-1, 6, bv);
  gp0.set_corner_vertex(k0, k1-1, 7, bv);
  gp1.set_corner_vertex(k0-1, k1-1, 2, bv);
  gp1.set_corner_vertex(k0, k1-1, 3, bv);
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
  gp1.set_corner_vertex(k0-1, k1-1, 2, bv);
  gp1.set_corner_vertex(k0-1, k1, 1, bv);
  gp1.set_corner_vertex(k0, k1-1, 3, bv);
  gp1.set_corner_vertex(k0, k1, 0, bv);
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
  gp0.set_corner_vertex(k0-1, k1-1, 6, bv);
  gp0.set_corner_vertex(k0-1, k1, 5, bv);
  gp0.set_corner_vertex(k0, k1-1, 7, bv);
  gp0.set_corner_vertex(k0, k1, 4, bv);
  return bv;
}

// ----------------------------------------------------------------------------
//
template <class Data_Type>
void CSurface<Data_Type>::make_triangles(Grid_Cell_List &gp0, Index k2)
{
  Stride step0 = stride[0], step1 = stride[1], step2 = stride[2];
  Index k0_size = size[0], k1_size = size[1], k2_size = size[2];
  Block_Array<Grid_Cell *> &clist = gp0.cells;
  Index cc = gp0.cell_count;
  const Data_Type *g0 = grid + (k2-1)*step2;
  Stride step01 = step0 + step1;
  for (Index k = 0 ; k < cc ; ++k)
    {
      Grid_Cell *c = clist.element(k);
      const Data_Type *gc = g0 + c->k0*step0 + c->k1*step1, *gc2 = gc + step2;
      int bits = ((gc[0] < threshold ? 0 : 1) |
		  (gc[step0] < threshold ? 0 : 2) |
		  (gc[step01] < threshold ? 0 : 4) |
		  (gc[step1] < threshold ? 0 : 8) |
		  (gc2[0] < threshold ? 0 : 16) |
		  (gc2[step0] < threshold ? 0 : 32) |
		  (gc2[step01] < threshold ? 0 : 64) |
		  (gc2[step1] < threshold ? 0 : 128));

      Index *cell_vertices = c->vertex;
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
  Index n3 = 3*vertex_count();
  for (Index v = 0 ; v < n3 ; v += 3)
    {  
      float x[3] = {vxyz.element(v), vxyz.element(v+1), vxyz.element(v+2)};
      float g[3];
      for (int a = 0 ; a < 3 ; ++a)
	g[a] = (x[a] == 0 ? 1 : (x[a] == size[a]-1 ? -1 : 0));
      if (g[0] == 0 && g[1] == 0 && g[2] == 0)
	{
	  Index i[3] = {(Index)x[0], (Index)x[1], (Index)x[2]};
	  const Data_Type *ga = grid + i[0]*stride[0]+i[1]*stride[1]+i[2]*stride[2];
	  const Data_Type *gb = ga;
	  Index off[3] = {0,0,0};
	  float fb = 0;
	  for (int a = 0 ; a < 3 ; ++a)
	    if ((fb = x[a]-i[a]) > 0) { off[a] = 1; gb = ga + stride[a]; break; }
	  float fa = 1-fb;
	  for (int a = 0 ; a < 3 ; ++a)
	    {
	      Stride s = stride[a];
	      Index ia = i[a], ib = ia + off[a];
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

// ----------------------------------------------------------------------------
//
template <class Data_Type>
Contour_Surface *surface(const Data_Type *grid, const Index size[3],
			 const Stride stride[3], float threshold, bool cap_faces)
{
  CSurface<Data_Type> *cs = new CSurface<Data_Type>(grid, size, stride,
						    threshold, cap_faces,
						    CONTOUR_ARRAY_BLOCK_SIZE);
  return cs;
}

} // end of namespace Contour_Calculation
