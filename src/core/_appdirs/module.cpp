// vi: set expandtab ts=4 sw=4:

#include <stdexcept>
#include "Python.h"

#include <appdirs/AppDirs.h>

extern "C" {

static PyObject*
init_paths(PyObject* /*self*/, PyObject* args)
{
    const char* path_sep;
    const char* user_data_dir;
    const char* user_config_dir;
    const char* user_cache_dir;
    const char* site_data_dir;
    const char* site_config_dir;
    const char* user_log_dir;
    const char* app_data_dir;

    if (!PyArg_ParseTuple(args, "ssssssss", &path_sep, &user_data_dir,
                &user_config_dir, &user_cache_dir, &site_data_dir,
                &site_config_dir, &user_log_dir, &app_data_dir))
        return NULL;
    try {
        appdirs::AppDirs::init_app_dirs(path_sep, user_data_dir,
                user_config_dir, user_cache_dir, site_data_dir,
                site_config_dir, user_log_dir, app_data_dir);
    } catch (std::logic_error &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
    Py_RETURN_NONE;
};


static const char* init_paths_doc =
"Initialize C++ app paths.  The eight arguments are strings.  The first string"
" is the character used to separate path name components and the next six"
" correspond to the following appdir module variables (in order):\n\n"

"user_data_dir\n"
"user_config_dir\n"
"user_cache_dir\n"
"site_data_dir\n"
"site_config_dir\n"
"user_log_dir\n\n"

"The final argument is the data/share path within the app itself\n";

static struct PyMethodDef appdirs_cpp_functions[] =
{
    {"init_paths", init_paths, METH_VARARGS, init_paths_doc },
    { NULL, NULL, 0, NULL }
};

static const char* mod_doc =
"The _appdirs module is used to inform the C++ layer about the file system"
" paths contained in the Python layer appdirs module object.";

static struct PyModuleDef appdirs_cpp_module =
{
    PyModuleDef_HEAD_INIT,
    "_appdirs",
    mod_doc,
    -1,
    appdirs_cpp_functions,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC
PyInit__appdirs()
{
    return PyModule_Create(&appdirs_cpp_module);
}

}  // extern "C"
