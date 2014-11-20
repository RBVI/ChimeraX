// vim: set expandtab ts=4 sw=4:

#include <iostream>

#include "Python.h"

namespace logger {

// placeholders until the Python Logger is implemented...
void  info(PyObject* logger, std::string& msg)
{
    std::cout << msg << "\n";
}

void  warning(PyObject* logger, std::string& msg)
{
    std::cout << "WARNING: " << msg << "\n";
}

void  error(PyObject* logger, std::string& msg)
{
    std::cerr << "ERROR: " << msg << "\n";
}


}
