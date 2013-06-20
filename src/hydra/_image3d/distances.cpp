// ----------------------------------------------------------------------------
//
#include <math.h>			// use sqrt()

namespace Distances
{

// ----------------------------------------------------------------------------
//
void distances_from_origin(float points[][3], int n, float origin[3],
			   float distances[])
{
  float x0 = origin[0], y0 = origin[1], z0 = origin[2];
  for (int k = 0 ; k < n ; ++k)
    {
      float *p = points[k];
      float dx = p[0] - x0, dy = p[1] - y0, dz = p[2] - z0;
      distances[k] = static_cast<float>(sqrt(dx*dx + dy*dy + dz*dz));
    }
}

// ----------------------------------------------------------------------------
//
void distances_perpendicular_to_axis(float points[][3], int n,
				     float origin[3], float axis[3],
				     float distances[])
{
  float x0 = origin[0], y0 = origin[1], z0 = origin[2];
  float ax = axis[0], ay = axis[1], az = axis[2];
  float norm = static_cast<float>(sqrt(ax*ax + ay*ay + az*az));
  if (norm != 0)
    { ax /= norm; ay /= norm ; ax /= norm; }
  for (int k = 0 ; k < n ; ++k)
    {
      float *p = points[k];
      float dx = p[0] - x0, dy = p[1] - y0, dz = p[2] - z0;
      float da = dx*ax + dy*ay + dz*az;
      float d2 = dx*dx + dy*dy + dz*dz - da*da;
      if (d2 < 0) d2 = 0;
      distances[k] = static_cast<float>(sqrt(d2));
    }
}

// ----------------------------------------------------------------------------
//
void distances_parallel_to_axis(float points[][3], int n,
				float origin[3], float axis[3],
				float distances[])
{
  float x0 = origin[0], y0 = origin[1], z0 = origin[2];
  float ax = axis[0], ay = axis[1], az = axis[2];
  float norm = static_cast<float>(sqrt(ax*ax + ay*ay + az*az));
  if (norm != 0)
    { ax /= norm; ay /= norm ; ax /= norm; }
  for (int k = 0 ; k < n ; ++k)
    {
      float *p = points[k];
      float dx = p[0] - x0, dy = p[1] - y0, dz = p[2] - z0;
      distances[k] = dx*ax + dy*ay + dz*az;
    }
}

// Return max(|Tp|)
float maximum_norm(float points[][3], int n, float t[3][4])
{
  float t00 = t[0][0], t01 = t[0][1], t02 = t[0][2], t03 = t[0][3];
  float t10 = t[1][0], t11 = t[1][1], t12 = t[1][2], t13 = t[1][3];
  float t20 = t[2][0], t21 = t[2][1], t22 = t[2][2], t23 = t[2][3];
  float d2max = 0;
  for (int k = 0 ; k < n ; ++k)
    {
      float *p = points[k];
      float x = p[0], y = p[1], z = p[2];
      float tpx = t00*x + t01*y + t02*z + t03;
      float tpy = t10*x + t11*y + t12*z + t13;
      float tpz = t20*x + t21*y + t22*z + t23;
      float d2 = tpx*tpx + tpy*tpy + tpz*tpz;
      if (d2 > d2max)
	d2max = d2;
    }
  float d = sqrt(d2max);
  return d;
}

// g = (|v-vm|^2*sum(wi*vi,j) - sum(wi*vi)*sum((vi-vm)*vi,j)) / |w||v-vm|^3
//   with vm = mean(v)
void correlation_gradient(float point_weights[], int n,
			  float values[], float gradients[][3],
			  bool about_mean, float *gradient_ret)
{
  double vm = 0, swv = 0, sw2 = 0, svvm2 = 0;
  double swg0 = 0, swg1 = 0, swg2 = 0;
  double svvmg0 = 0, svvmg1 = 0, svvmg2 = 0;
  if (about_mean)
    {
      for (int k = 0 ; k < n ; ++k)
	vm += values[k];
      if (n > 0)
	vm /= n;
    }
  for (int k = 0 ; k < n ; ++k)
    {
      float w = point_weights[k], v = values[k], *g = gradients[k];
      float g0 = g[0], g1 = g[1], g2 = g[2];
      float vvm = v - vm;
      svvm2 += vvm*vvm;
      swv += w*v;
      sw2 += w*w;
      swg0 += w*g0; swg1 += w*g1; swg2 += w*g2;
      svvmg0 += vvm*g0; svvmg1 += vvm*g1; svvmg2 += vvm*g2;
    }
  double cg0 = svvm2*swg0 - swv*svvmg0;
  double cg1 = svvm2*swg1 - swv*svvmg1;
  double cg2 = svvm2*swg2 - swv*svvmg2;
  double nvvm = sqrt(svvm2), nw = sqrt(sw2);
  double nvw = nvvm*nvvm*nvvm*nw;
  if (nvw > 0)
    {
      cg0 /= nvw; cg1 /= nvw; cg2 /= nvw;
    }
  gradient_ret[0] = cg0;
  gradient_ret[1] = cg1;
  gradient_ret[2] = cg2;
}


// Return sum(w*(p-c) x f).
void torque(float points[][3], int n, float *point_weights,
	    float forces[][3], float center[3], float *torque_ret)
{
  float c0 = center[0], c1 = center[1], c2 = center[2];
  double t0 = 0, t1 = 0, t2 = 0;
  for (int k = 0 ; k < n ; ++k)
    {
      float *p = points[k];
      float pc0 = p[0]-c0, pc1 = p[1]-c1, pc2 = p[2]-c2;
      float w = (point_weights ? point_weights[k] : 1.0);
      float *f = forces[k];
      float f0 = f[0], f1 = f[1], f2 = f[2];
      t0 += w*(pc1*f2 - pc2*f1);
      t1 += w*(pc2*f0 - pc0*f2);
      t2 += w*(pc0*f1 - pc1*f0);
    }
  torque_ret[0] = t0;
  torque_ret[1] = t1;
  torque_ret[2] = t2;
}

// Return (p-c) x f.
void torques(float points[][3], int n, float center[3], float forces[][3],
	     float torques[][3])
{
  float c0 = center[0], c1 = center[1], c2 = center[2];
  for (int k = 0 ; k < n ; ++k)
    {
      float *p = points[k], *f = forces[k], *t = torques[k];
      float pc0 = p[0]-c0, pc1 = p[1]-c1, pc2 = p[2]-c2;
      float f0 = f[0], f1 = f[1], f2 = f[2];
      t[0] = pc1*f2 - pc2*f1;
      t[1] = pc2*f0 - pc0*f2;
      t[2] = pc0*f1 - pc1*f0;
    }
}

// Return (|v-vm|^2*sum(w*(rxg)) - sum(w*v)*sum((v-vm)*(rxg)) / |v-vm|^3
//   with r = p-c, vm = mean(v)
void correlation_torque(float points[][3], int n, float point_weights[],
			float values[], float gradients[][3], float center[3],
			bool about_mean, float *torque_ret)
{
  float c0 = center[0], c1 = center[1], c2 = center[2];
  double vm = 0, swv = 0, sw2 = 0, svvm2 = 0;
  double wrg0 = 0, wrg1 = 0, wrg2 = 0;
  double vvmrg0 = 0, vvmrg1 = 0, vvmrg2 = 0;
  if (about_mean)
    {
      for (int k = 0 ; k < n ; ++k)
	vm += values[k];
      if (n > 0)
	vm /= n;
    }

  for (int k = 0 ; k < n ; ++k)
    {
      float *p = points[k];
      float r0 = p[0] - c0, r1 = p[1] - c1, r2 = p[2] - c2;
      float *g = gradients[k];
      float g0 = g[0], g1 = g[1], g2 = g[2];
      float rg0 = r1*g2-r2*g1, rg1 = r2*g0-r0*g2, rg2 = r0*g1-r1*g0;
      float v = values[k];
      float vvm = v - vm;
      svvm2 += vvm*vvm;
      float w = point_weights[k];
      swv += w*v;
      sw2 += w*w;
      wrg0 += w*rg0; wrg1 += w*rg1; wrg2 += w*rg2;
      vvmrg0 += vvm*rg0; vvmrg1 += vvm*rg1; vvmrg2 += vvm*rg2;
    }
  double t0 = svvm2*wrg0 - swv*vvmrg0;
  double t1 = svvm2*wrg1 - swv*vvmrg1;
  double t2 = svvm2*wrg2 - swv*vvmrg2;
  double nvvm = sqrt(svvm2), nw = sqrt(sw2);
  double nvw = nvvm*nvvm*nvvm*nw;
  if (nvw > 0)
    {
      t0 /= nvw; t1 /= nvw; t2 /= nvw;
    }
  torque_ret[0] = t0;
  torque_ret[1] = t1;
  torque_ret[2] = t2;
}

// Return (|v-vm|^2*sum(w*(rxg)) - sum(w*v)*sum((v-vm)*(rxg)) / |v-vm|^3
//   with vm = mean(v)
void correlation_torque2(float point_weights[], int n,
			 float values[], float rxg[][3],
			 bool about_mean, float *torque_ret)
{
  double vm = 0, swv = 0, sw2 = 0, svvm2 = 0;
  double wrg0 = 0, wrg1 = 0, wrg2 = 0;
  double vvmrg0 = 0, vvmrg1 = 0, vvmrg2 = 0;
  if (about_mean)
    {
      for (int k = 0 ; k < n ; ++k)
	vm += values[k];
      if (n > 0)
	vm /= n;
    }

  for (int k = 0 ; k < n ; ++k)
    {
      float v = values[k], w = point_weights[k], *rg = rxg[k];
      float rg0 = rg[0], rg1 = rg[1], rg2 = rg[2];
      float vvm = v - vm;
      svvm2 += vvm*vvm;
      swv += w*v;
      sw2 += w*w;
      wrg0 += w*rg0; wrg1 += w*rg1; wrg2 += w*rg2;
      vvmrg0 += vvm*rg0; vvmrg1 += vvm*rg1; vvmrg2 += vvm*rg2;
    }
  double t0 = svvm2*wrg0 - swv*vvmrg0;
  double t1 = svvm2*wrg1 - swv*vvmrg1;
  double t2 = svvm2*wrg2 - swv*vvmrg2;
  double nvvm = sqrt(svvm2), nw = sqrt(sw2);
  double nvw = nvvm*nvvm*nvvm*nw;
  if (nvw > 0)
    {
      t0 /= nvw; t1 /= nvw; t2 /= nvw;
    }
  torque_ret[0] = t0;
  torque_ret[1] = t1;
  torque_ret[2] = t2;
}


}  // end of namespace Distances
