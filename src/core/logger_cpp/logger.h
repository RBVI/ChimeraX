// vi: set expandtab ts=4 sw=4:
#ifndef cpp_logger_Logger
#define cpp_logger_Logger

#include <sstream>
#include <string>

// "forward declare" PyObject, which is a typedef of a struct,
// as per the python mailing list:
// http://mail.python.org/pipermail/python-dev/2003-August/037601.html
#ifndef PyObject_HEAD
struct _object;
typedef _object PyObject;
#endif
    
namespace logger {

enum class _LogLevel { INFO, WARNING, ERROR };

void  _log(PyObject* logger, std::stringstream& msg, _LogLevel level);
template<typename T, typename... Args>
void  _log(PyObject* logger, std::stringstream& msg, _LogLevel level,
    T value, Args... args)
{
    msg << value;
    _log(logger, msg, level, args...);
}

// 'logger' arg can be nullptr

template<typename T, typename... Args>
void  info(PyObject* logger, T value, Args... args)
{
    std::stringstream msg;
    msg << value;
    _log(logger, msg, _LogLevel::INFO, args...);
}

template<typename T, typename... Args>
void  warning(PyObject* logger, T value, Args... args)
{
    std::stringstream msg;
    msg << value;
    _log(logger, msg, _LogLevel::WARNING, args...);
}

template<typename T, typename... Args>
void  error(PyObject* logger, T value, Args... args)
{
    std::stringstream msg;
    msg << value;
    _log(logger, msg, _LogLevel::ERROR, args...);
}

} //  namespace logger

#endif  // cpp_logger_Logger
