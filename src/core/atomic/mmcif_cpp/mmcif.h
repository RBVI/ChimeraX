// vi: set expandtab ts=4 sw=4:
#include <Python.h>
#include <string>
#include <functional>
#include <vector>

#include <atomstruct/string_types.h>

namespace tmpl {
    class Residue;
}

namespace mmcif {

using atomstruct::ResName;

PyObject*   parse_mmCIF_file(const char* filename,
                            PyObject* change_tracker_ptr, PyObject* logger);
PyObject*   parse_mmCIF_file(const char* filename,
                             const std::vector<std::string> &extra_categories,
                             PyObject* change_tracker_ptr, PyObject* logger);
PyObject*   parse_mmCIF_buffer(const unsigned char* buffer,
                             PyObject* change_tracker_ptr, PyObject* logger);
PyObject*   parse_mmCIF_buffer(const unsigned char* buffer,
                             const std::vector<std::string> &extra_categories,
                             PyObject* change_tracker_ptr, PyObject* logger);
void        load_mmCIF_templates(const char* filename);
void        set_Python_locate_function(PyObject* function);

#ifndef WrapPy
const tmpl::Residue*
            find_template_residue(const ResName& name);
typedef std::function<std::string (const ResName& residue_type)>
            LocateFunc;
void        set_locate_template_function(LocateFunc func);
#endif

}  // namespace mmcif
