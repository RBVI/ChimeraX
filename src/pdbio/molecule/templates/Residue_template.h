// vim: set expandtab ts=4 sw=4:
#ifndef templates_Residue_template
#define templates_Residue_template

#include <string>

class TmplAtom;

// functor to convert the two types of atom assignment methods
class tmpl_assigner : public std::binary_function<TmplAtom *, const char *, void> {
    void (TmplAtom::*a_func)(const char *);
    void (*func)(TmplAtom *, const char *);
public:
    tmpl_assigner(void (TmplAtom::*assign_func)(const char *)) :
                        a_func(assign_func), func(NULL) {}
    tmpl_assigner(void (*assign_func)(TmplAtom *, const char *)) :
                        a_func(NULL), func(assign_func) {}
    void operator()(TmplAtom *a, std::string &val) {
        if (a_func)
            (a->*a_func)(val.c_str());
        else
            (*func)(a, val.c_str());
    }
};

#endif  // templates_Residue_template
