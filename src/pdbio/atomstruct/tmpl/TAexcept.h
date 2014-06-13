// vim: set expandtab ts=4 sw=4:
#ifndef templates_TAexcept
#define templates_TAexcept

#include <stdexcept>
#include "../imex.h"

// things templateAssign() can throw...
class ATOMSTRUCT_IMEX TA_exception : public std::runtime_error {
public:
    TA_exception(const std::string &msg) : std::runtime_error(msg) {}
};
class ATOMSTRUCT_IMEX TA_TemplateSyntax : public TA_exception {
public:
    TA_TemplateSyntax(const std::string &msg) : TA_exception(msg) {}
};
class ATOMSTRUCT_IMEX TA_NoTemplate : public TA_exception {
public:
    TA_NoTemplate(const std::string &msg) : TA_exception(msg) {}
};

#endif  // templates_TAexcept
