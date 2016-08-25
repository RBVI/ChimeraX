// vi: set expandtab ts=4 sw=4:

/*
 * === UCSF ChimeraX Copyright ===
 * Copyright 2016 Regents of the University of California.
 * All rights reserved.  This software provided pursuant to a
 * license agreement containing restrictions on its disclosure,
 * duplication and use.  For details see:
 * http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
 * This notice must be embedded in or attached to all copies,
 * including partial copies, of the software or any revisions
 * or derivations thereof.
 * === UCSF ChimeraX Copyright ===
 */

#ifndef cpp_logger_Logger
#define cpp_logger_Logger

#include <sstream>
#include <string>

#include "imex.h"

// "forward declare" PyObject, which is a typedef of a struct,
// as per the python mailing list:
// http://mail.python.org/pipermail/python-dev/2003-August/037601.html
#ifndef PyObject_HEAD
struct _object;
typedef _object PyObject;
#endif
    
namespace logger {

enum class _LogLevel { INFO, WARNING, ERROR };

LOGGER_IMEX
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
