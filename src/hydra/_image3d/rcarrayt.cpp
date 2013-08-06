// ----------------------------------------------------------------------------
//
template <class T>
class Array_Operator
{
 public:
  virtual ~Array_Operator() {}
  virtual T operator()(T x) = 0;
};

// ----------------------------------------------------------------------------
//
template <class T>
Array<T>::Array() : Untyped_Array()
{
}

// ----------------------------------------------------------------------------
//
template <class T>
Array<T>::Array(int dim, const int *size) : Untyped_Array(sizeof(T), dim, size)
{
}

// ----------------------------------------------------------------------------
//
template <class T>
Array<T>::Array(int dim, const int *size, const T *values) :
  Untyped_Array(sizeof(T), dim, size, (void *)values)
{
}

// ----------------------------------------------------------------------------
//
template <class T>
Array<T>::Array(int dim, const int *size, T *values, Release_Data *release) :
  Untyped_Array(sizeof(T), dim, size, (void *)values, release)
{
}

// ----------------------------------------------------------------------------
//
template <class T>
Array<T>::Array(const Untyped_Array &a) : Untyped_Array(a)
{
}

// ----------------------------------------------------------------------------
//
template <class T>
Array<T>::Array(const Array<T> &a) : Untyped_Array(a)
{
}

// ----------------------------------------------------------------------------
//
template <class T>
const Array<T> &Array<T>::operator=(const Array<T> &array)
{
  Untyped_Array::operator=(array);
  return *this;
}

// ----------------------------------------------------------------------------
//
template <class T>
Array<T>::~Array()
{
}

// ----------------------------------------------------------------------------
//
template <class T>
T Array<T>::value(const int *index) const
{
  long i = 0;
  int dim = dimension();
  for (int a = 0 ; a < dim ; ++a)
    i += index[a] * stride(a);
  T *data = values();
  return data[i];
}

// ----------------------------------------------------------------------------
//
template <class T>
T *Array<T>::values() const
{
  return (T *) Untyped_Array::values();
}

// ----------------------------------------------------------------------------
//
template <class T>
T *Array<T>::copy_of_values() const
{
  long s = size();
  T *v = new T[s];
  get_values(v);
  return v;
}

// ----------------------------------------------------------------------------
// Optimized for dimension <= 4.
//
template <class T>
void Array<T>::get_values(T *v) const
{
  if (is_contiguous())
    {
      T *d = values();
      long length = size();
      for (long i = 0 ; i < length ; ++i)
	v[i] = d[i];
      return;
    }

  if (dimension() == 0)
    return;

  T *d = values();

  long k = 0;
  long i0, j0, js0 = stride(0), s0 = size(0);
  if (dimension() == 1)
    {
      for (i0=0, j0=0 ; i0<s0 ; ++i0, j0+=js0)
	v[k++] = d[j0];
      return;
    }

  long i1, j1, js1 = stride(1), s1 = size(1);
  if (dimension() == 2)
    {
      for (i0=0, j0=0 ; i0<s0 ; ++i0, j0+=js0)
	for (i1=0, j1=j0 ; i1<s1 ; ++i1, j1+=js1)
	  v[k++] = d[j1];
      return;
    }

  long i2, j2, js2 = stride(2), s2 = size(2);
  if (dimension() == 3)
    {
      for (i0=0, j0=0 ; i0<s0 ; ++i0, j0+=js0)
	for (i1=0, j1=j0 ; i1<s1 ; ++i1, j1+=js1)
	  for (i2=0, j2=j1 ; i2<s2 ; ++i2, j2+=js2)
	    v[k++] = d[j2];
      return;
    }

  long i3, j3, js3 = stride(3), s3 = size(3);
  if (dimension() == 4)
    {
      for (i0=0, j0=0 ; i0<s0 ; ++i0, j0+=js0)
	for (i1=0, j1=j0 ; i1<s1 ; ++i1, j1+=js1)
	  for (i2=0, j2=j1 ; i2<s2 ; ++i2, j2+=js2)
	    for (i3=0, j3=j2 ; i3<s3 ; ++i3, j3+=js3)
	      v[k++] = d[j3];
      return;
    }

  long step = size()/size(0);
  for (long i = 0 ; i < s0 ; ++i)
    slice(0,i).get_values(v + i*step);
}

// ----------------------------------------------------------------------------
//
template <class T>
void Array<T>::set(const int *index, T value)
{
  long i = 0;
  int dim = dimension();
  for (int a = 0 ; a < dim ; ++a)
    i += index[a] * stride(a);
  T *data = values();
  data[i] = value;
}

// ----------------------------------------------------------------------------
// Optimized for contiguous arrays and dimension <= 4.
//
template <class T>
void Array<T>::set(T value)
{
  if (is_contiguous())
    {
      T *d = values();
      long length = size();
      for (int i = 0 ; i < length ; ++i)
	d[i] = value;
      return;
    }

  if (dimension() == 0)
    return;

  T *d = values();

  long i0, j0, js0 = stride(0), s0 = size(0);
  if (dimension() == 1)
    {
      for (i0=0, j0=0 ; i0<s0 ; ++i0, j0+=js0)
	d[j0] = value;
      return;
    }

  long i1, j1, js1 = stride(1), s1 = size(1);
  if (dimension() == 2)
    {
      for (i0=0, j0=0 ; i0<s0 ; ++i0, j0+=js0)
	for (i1=0, j1=j0 ; i1<s1 ; ++i1, j1+=js1)
	  d[j1] = value;
      return;
    }

  long i2, j2, js2 = stride(2), s2 = size(2);
  if (dimension() == 3)
    {
      for (i0=0, j0=0 ; i0<s0 ; ++i0, j0+=js0)
	for (i1=0, j1=j0 ; i1<s1 ; ++i1, j1+=js1)
	  for (i2=0, j2=j1 ; i2<s2 ; ++i2, j2+=js2)
	    d[j2] = value;
      return;
    }

  long i3, j3, js3 = stride(3), s3 = size(3);
  if (dimension() == 4)
    {
      for (i0=0, j0=0 ; i0<s0 ; ++i0, j0+=js0)
	for (i1=0, j1=j0 ; i1<s1 ; ++i1, j1+=js1)
	  for (i2=0, j2=j1 ; i2<s2 ; ++i2, j2+=js2)
	    for (i3=0, j3=j2 ; i3<s3 ; ++i3, j3+=js3)
	      d[j3] = value;
      return;
    }

  for (long i = 0 ; i < s0 ; ++i)
    this->slice(0,i).set(value);
}

// ----------------------------------------------------------------------------
// Optimized for dimension <= 4.
//
template <class T>
template <class S>
void Array<T>::set(const Array<S> &a)
{
  if (a.dimension() != dimension())
    throw std::invalid_argument("void Array::set(): dimension mismatch");

  if (dimension() == 0)
    return;

  T *d = values();
  S *ad = a.values();
  
  long i0, j0, js0 = stride(0), k0, ks0 = a.stride(0);
  int s0 = (size(0) < a.size(0) ? size(0) : a.size(0));
  if (dimension() == 1)
    {
      for (i0=0, j0=0, k0=0 ; i0<s0 ; ++i0, j0+=js0, k0+=ks0)
	d[j0] = static_cast<T>(ad[k0]);
      return;
    }


  long i1, j1, js1 = stride(1), k1, ks1 = a.stride(1);
  int s1 = (size(1) < a.size(1) ? size(1) : a.size(1));
  if (dimension() == 2)
    {
      for (i0=0, j0=0, k0=0 ; i0<s0 ; ++i0, j0+=js0, k0+=ks0)
	for (i1=0, j1=j0, k1=k0 ; i1<s1 ; ++i1, j1+=js1, k1+=ks1)
	  d[j1] = static_cast<T>(ad[k1]);
      return;
    }

  long i2, j2, js2 = stride(2), k2, ks2 = a.stride(2);
  int s2 = (size(2) < a.size(2) ? size(2) : a.size(2));
  if (dimension() == 3)
    {
      for (i0=0, j0=0, k0=0 ; i0<s0 ; ++i0, j0+=js0, k0+=ks0)
	for (i1=0, j1=j0, k1=k0 ; i1<s1 ; ++i1, j1+=js1, k1+=ks1)
	  for (i2=0, j2=j1, k2=k1 ; i2<s2 ; ++i2, j2+=js2, k2+=ks2)
	    d[j2] = static_cast<T>(ad[k2]);
      return;
    }

  long i3, j3, js3 = stride(3), k3, ks3 = a.stride(3);
  int s3 = (size(3) < a.size(3) ? size(3) : a.size(3));
  if (dimension() == 4)
    {
      for (i0=0, j0=0, k0=0 ; i0<s0 ; ++i0, j0+=js0, k0+=ks0)
	for (i1=0, j1=j0, k1=k0 ; i1<s1 ; ++i1, j1+=js1, k1+=ks1)
	  for (i2=0, j2=j1, k2=k1 ; i2<s2 ; ++i2, j2+=js2, k2+=ks2)
	    for (i3=0, j3=j2, k3=k2 ; i3<s3 ; ++i3, j3+=js3, k3+=ks3)
	      d[j3] = static_cast<T>(ad[k3]);
      return;
    }

  for (long i = 0 ; i < s0 ; ++i)
    this->slice(0,i).set(a.slice(0,i));
}

// ----------------------------------------------------------------------------
// Optimized for dimension <= 4.
//
template <class T>
void Array<T>::apply(Array_Operator<T> &op)
{
  if (dimension() == 0)
    return;

  T *d = values();
  
  long i0, j0, js0 = stride(0), s0 = size(0);
  if (dimension() == 1)
    {
      for (i0=0, j0=0 ; i0<s0 ; ++i0, j0+=js0)
	d[j0] = op(d[j0]);
      return;
    }

  long i1, j1, js1 = stride(1), s1 = size(1);
  if (dimension() == 2)
    {
      for (i0=0, j0=0 ; i0<s0 ; ++i0, j0+=js0)
	for (i1=0, j1=j0 ; i1<s1 ; ++i1, j1+=js1)
	  d[j1] = op(d[j1]);
      return;
    }

  long i2, j2, js2 = stride(2), s2 = size(2);
  if (dimension() == 3)
    {
      for (i0=0, j0=0 ; i0<s0 ; ++i0, j0+=js0)
	for (i1=0, j1=j0 ; i1<s1 ; ++i1, j1+=js1)
	  for (i2=0, j2=j1 ; i2<s2 ; ++i2, j2+=js2)
	    d[j2] = op(d[j2]);
      return;
    }

  long i3, j3, js3 = stride(3), s3 = size(3);
  if (dimension() == 4)
    {
      for (i0=0, j0=0 ; i0<s0 ; ++i0, j0+=js0)
	for (i1=0, j1=j0 ; i1<s1 ; ++i1, j1+=js1)
	  for (i2=0, j2=j1 ; i2<s2 ; ++i2, j2+=js2)
	    for (i3=0, j3=j2 ; i3<s3 ; ++i3, j3+=js3)
	      d[j3] = op(d[j3]);
      return;
    }

  for (long i = 0 ; i < s0 ; ++i)
    this->slice(0,i).apply(op);
}

// ----------------------------------------------------------------------------
//
template <class T>
Array<T> Array<T>::slice(int axis, int index) const
{
  return Array<T>(Untyped_Array::slice(axis, index));
}

// ----------------------------------------------------------------------------
//
template <class T>
Array<T> Array<T>::subarray(int axis, int i_min, int i_max)
{
  return Array<T>(Untyped_Array::subarray(axis, i_min, i_max));
}

// ----------------------------------------------------------------------------
//
template <class T>
Array<T> Array<T>::contiguous_array() const
{
  if (is_contiguous())
    return *this;

  return this->copy();
}

// ----------------------------------------------------------------------------
//
template <class T>
Array<T> Array<T>::copy() const
{
  T *v = copy_of_values();
  return Array<T>(dimension(), sizes(), v, new Delete_Data<T>(v));
}
