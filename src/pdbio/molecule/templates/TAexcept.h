#ifndef templates_TAexcept
#define templates_TAexcept

#include <stdexcept>

// things templateAssign() can throw...
class TA_exception : public std::runtime_error {
public:
	TA_exception(const std::string &msg) : std::runtime_error(msg) {}
};
class TA_TemplateSyntax : public TA_exception {
public:
	TA_TemplateSyntax(const std::string &msg) : TA_exception(msg) {}
};
class TA_NoTemplate : public TA_exception {
public:
	TA_NoTemplate(const std::string &msg) : TA_exception(msg) {}
};

#endif  // templates_TAexcept
