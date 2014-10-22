// vim: set expandtab ts=4 sw=4:

#include <stdexcept>
#include "Python.h"

#include "AppDirs.h"

extern "C" {

static PyObject*
init_paths(PyObject* self, PyObject* args)
{
    const char* user_data_dir;
    const char* user_config_dir;
    const char* user_cache_dir;
    const char* site_data_dir;
    const char* site_config_dir;
    const char* user_log_dir;

    if (!PyArg_ParseTuple(args, "ssssss", &user_data_dir, &user_config_dir,
            &user_cache_dir, &site_data_dir, &site_config_dir, &user_log_dir))
        return NULL;
    try {
        cpp_appdirs::AppDirs::init_app_dirs(user_data_dir, user_config_dir,
            user_cache_dir, site_data_dir, site_config_dir, user_log_dir);
    } catch (std::logic_error &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
    Py_RETURN_NONE;
};


static const char* init_paths_doc =
"Initialize C++ app paths.  The six arguments are strings that correspond"
" to the following appdir module variables (in order):\n\n"

"user_data_dir\n"
"user_config_dir\n"
"user_cache_dir\n"
"site_data_dir\n"
"site_config_dir\n"
"user_log_dir\n";

static struct PyMethodDef cpp_appdirs_functions[] =
{
    {"init_paths", init_paths, METH_VARARGS, init_paths_doc },
    { NULL, NULL, 0, NULL }
};

static const char* mod_doc =
"The cpp_addirs module is used to inform the C++ layer about the file system"
" paths contained in the Python layer appdirs module object.";

static struct PyModuleDef cpp_appdirs_module =
{
    PyModuleDef_HEAD_INIT,
    "cpp_appdirs",
    mod_doc,
    -1,
    cpp_appdirs_functions,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit_cpp_appdirs()
{
    return PyModule_Create(&cpp_appdirs_module);
}

}  // extern "C"
