// vi: set expandtab shiftwidth=4 softtabstop=4:
// ----------------------------------------------------------------------------
//
#ifndef REFCOUNT_HEADER_INCLUDED
#define REFCOUNT_HEADER_INCLUDED

class Reference_Count
{
public:
  Reference_Count();
  Reference_Count(const Reference_Count &);
  Reference_Count &operator=(const Reference_Count &);
  virtual ~Reference_Count();
  int reference_count() const;
private:
  int *ref_count;
};

#endif
