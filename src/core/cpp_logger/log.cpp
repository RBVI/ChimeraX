// vim: set expandtab ts=4 sw=4:

#include <iostream>

#include "Python.h"
#include "logger.h"

namespace logger {

// placeholders until the Python Logger is implemented...
void  _log(PyObject* logger, std::stringstream& msg, _LogLevel level)
{
    if (logger == nullptr) {
        if (level == _LogLevel::ERROR)
            std::cerr << msg.str() << "\n";
        else
            std::cout << msg.str() << "\n";
    } else {
        // Python logging goes here
    }
}

} // namespace logger
