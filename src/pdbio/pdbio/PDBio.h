// vim: set expandtab ts=4 sw=4:
#include <Python.h>

namespace pdb {
	
extern "C" {
PyObject * read_pdb_file(PyObject *, PyObject *args, PyObject *keywords);
PyObject * read_pdb(PyObject *pdb_file, PyObject *log_file, bool explode);
PyObject * parse_mmCIF(const char *whole_file, PyObject *log_file, bool explode);
}

}  // namespace pdb
