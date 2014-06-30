//
// Inlined string to float/integer conversions are more than twice as fast as standard library strtof, strtol.
//
#ifndef STRINGNUM_HEADER_INCLUDED
#define STRINGNUM_HEADER_INCLUDED

inline float string_to_float(const char *s)
{
  float fa = 0, v = 0;
  bool neg = (*s == '-');
  if (neg)
    s += 1;
  while (true)
    {
      char c = *s;
      if (c >= '0' && c <= '9')
	{
	  if (fa)
	    { v += fa * (c-'0'); fa *= 0.1; }
	  else
	    v = 10*v + (c-'0');
	}
      else if (c == '.')
	fa = 0.1;
      else
	break;
      s += 1;
    }
  return (neg ? -v : v);
}

inline int string_to_integer(const char *s)
{
  bool neg = (*s == '-');
  int v = 0;
  if (neg)
    s += 1;
  
  while (true)
    {
      char c = *s;
      if (c >= '0' && c <= '9')
	v = 10*v + (c-'0');
      else
	break;
      s += 1;
    }
  return (neg ? -v : v);
}

#endif
