// ----------------------------------------------------------------------------
// Routines to calculate distances from arrays of points to lines or an origin.
//
#ifndef DISTANCES_HEADER_INCLUDED
#define DISTANCES_HEADER_INCLUDED

namespace Distances
{

void distances_from_origin(float points[][3], int n, float origin[3],
			   float distances[]);
void distances_perpendicular_to_axis(float points[][3], int n,
				     float origin[3], float axis[3],
				     float distances[]);
void distances_parallel_to_axis(float points[][3], int n,
				float origin[3], float axis[3],
				float distances[]);

float maximum_norm(float points[][3], int n, float tf[3][4]);
void correlation_gradient(float point_weights[], int n,
			  float values[], float gradients[][3],
			  bool about_mean, float *gradient_ret);
void torque(float points[][3], int n, float *point_weights,
	    float forces[][3], float center[3], float *torque_ret);
void torques(float points[][3], int n, float center[3], float forces[][3],
	     float torques[][3]);
void correlation_torque(float points[][3], int n, float point_weights[],
			float values[], float gradients[][3], float center[3],
			bool about_mean, float *torque_ret);
void correlation_torque2(float point_weights[], int n,
			 float values[], float rxg[][3],
			 bool about_mean, float *torque_ret);
	     

}  // end of namespace Distances

#endif
