// vi: set expandtab ts=4 sw=4:

/*
 * === UCSF ChimeraX Copyright ===
 * Copyright 2022 Regents of the University of California. All rights reserved.
 * This software is provided pursuant to the ChimeraX license agreement, which
 * covers academic and commercial uses. For more information, see
 * <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
 *
 * This file is part of the ChimeraX library. You can also redistribute and/or
 * modify it under the GNU Lesser General Public License version 2.1 as
 * published by the Free Software Foundation. For more details, see
 * <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
 *
 * This file is distributed WITHOUT ANY WARRANTY; without even the implied
 * warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
 * must be embedded in or attached to all copies, including partial copies, of
 * the software or any revisions or derivations thereof.
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

#ifdef _WIN32
# undef INFO
# undef WARNING
# undef ERROR
#endif
enum class _LogLevel { INFO, WARNING, ERROR };

void  _log(PyObject* logger, std::stringstream& msg, _LogLevel level, bool is_html=false);
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

inline void  _html_log(PyObject* logger, std::stringstream& msg, _LogLevel level) {
    _log(logger, msg, level, true);
}

template<typename T, typename... Args>
void  _html_log(PyObject* logger, std::stringstream& msg, _LogLevel level,
    T value, Args... args)
{
    msg << value;
    _html_log(logger, msg, level, args...);
}

// 'logger' arg can be nullptr

template<typename T, typename... Args>
void  html_info(PyObject* logger, T value, Args... args)
{
    std::stringstream msg;
    msg << value;
    _html_log(logger, msg, _LogLevel::INFO, args...);
}

template<typename T, typename... Args>
void  html_warning(PyObject* logger, T value, Args... args)
{
    std::stringstream msg;
    msg << value;
    _html_log(logger, msg, _LogLevel::WARNING, args...);
}

template<typename T, typename... Args>
void  html_error(PyObject* logger, T value, Args... args)
{
    std::stringstream msg;
    msg << value;
    _html_log(logger, msg, _LogLevel::ERROR, args...);
}

} //  namespace logger

#endif  // cpp_logger_Logger
