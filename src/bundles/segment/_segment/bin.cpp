// ----------------------------------------------------------------------------
//

namespace Segmentation_Calculation
{

void bin_sums(float *xyz, int n, float *v, float b0, float bsize, int bcount,
	      float *bsums, int *bcounts)
{
  float vx = v[0], vy = v[1], vz = v[2];
  for (int i = 0 ; i < n ; ++i)
    {
      float *p = &xyz[3*i];
      float px = p[0], py = p[1], pz = p[2];
      float pv = px*vx + py*vy + pz*vz;
      int b = int((pv - b0) / bsize);
      if (b >= 0 && b < bcount)
	{
	  bcounts[b] += 1;
	  float *bs = &bsums[3*b];
	  bs[0] += px; bs[1] += py; bs[2] += pz;
	}
    }
}


} // end of namespace Segmentation_Calculation
