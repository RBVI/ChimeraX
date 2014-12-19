#include <Python.h>
#include <sys/types.h>
#include <sys/sysctl.h>

// Returns the size of physical memory (RAM) in bytes.

extern "C" PyObject *memory_size(PyObject *s, PyObject *args, PyObject *keywds)
{
  const char *kwlist[] = {NULL};
  if (!PyArg_ParseTupleAndKeywords(args, keywds, const_cast<char *>(""), (char **)kwlist))
    return NULL;

  int mib[2];
  mib[0] = CTL_HW;
  mib[1] = HW_MEMSIZE;

  int64_t size = 0;               /* 64-bit */
  size_t len = sizeof( size );
  if ( sysctl( mib, 2, &size, &len, NULL, 0 ) == 0 )
    return PyLong_FromSize_t((size_t)size);
  return PyLong_FromSize_t(0);
}
