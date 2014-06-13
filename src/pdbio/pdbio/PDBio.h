// vim: set expandtab ts=4 sw=4:
#include <Python.h>

namespace pdb {
	
PyObject * read_pdb_file(PyObject *, PyObject *args, PyObject *keywords);
PyObject * read_pdb(PyObject *pdb_file, PyObject *log_file, bool explode);

}  // namespace pdb
