#include <Python.h>

PyObject * read_pdb_file(PyObject *, PyObject *args, PyObject *keywords);
PyObject * read_pdb(PyObject *pdb_file, PyObject *log_file, bool explode);
