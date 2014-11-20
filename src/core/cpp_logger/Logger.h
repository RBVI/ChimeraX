// vim: set expandtab ts=4 sw=4:
#ifndef cpp_logger_Logger
#define cpp_logger_Logger

#include <string>

namespace logger {

class PyObject;
    
void  info(PyObject* logger, std::string& msg);
void  warning(PyObject* logger, std::string& msg);
void  error(PyObject* logger, std::string& msg);

} //  namespace logger

#endif  // cpp_logger_Logger
