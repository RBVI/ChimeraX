// vim: set expandtab ts=4 sw=4:
#include <Python.h>

namespace mmcif {
	
PyObject*   parse_mmCIF_file(const char *filename);
PyObject*   parse_mmCIF_buffer(const unsigned char *buffer);
void        load_mmCIF_templates(const char *filename, const char *category=NULL);

#ifndef WrapPy
bool        init_structaccess();
#endif

}  // namespace mmcif
