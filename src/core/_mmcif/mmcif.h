// vi: set expandtab ts=4 sw=4:
#include <Python.h>
#include <string>
#include <functional>

namespace tmpl {
    class Residue;
}

namespace mmcif {

PyObject*   parse_mmCIF_file(const char* filename);
PyObject*   parse_mmCIF_buffer(const unsigned char* buffer);
void        load_mmCIF_templates(const char* filename);
void        set_Python_locate_function(PyObject* function=NULL);

#ifndef WrapPy
const tmpl::Residue*
            find_template_residue(const std::string& name);
typedef std::function<std::string (const std::string& residue_type)>
            LocateFunc;
void        set_locate_template_function(LocateFunc func);
bool        init_structaccess();
#endif

}  // namespace mmcif
