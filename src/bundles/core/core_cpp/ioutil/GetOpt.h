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

#ifndef ioutil_GetOpt_h
# define ioutil_GetOpt_h

namespace ioutil {

class GetOpt {
public:
    GetOpt(int argc, char *const *argv, char const *opts,
                        bool printErrors = true);
    int     index() const;  // arg index of option
    char const  *arg() const;   // optional argument to option
    int     option() const; // option being processed
    void        reset();    // restart processing arguments
    void        skip(int count);    // skip count args
    void        printErrors(bool b);    // print errors on stderr?
    int     operator()();   // return next option
private:
    bool        opterr;     // if error message should be printed
    int     optind;     // index into parent argv vector
    int     optopt;     // character checked for validity
    bool        optreset;   // reset getopt
    char const  *optarg;    // argument associated with option
    int     nargc;      // saved argument count
    char *const *nargv;     // saved argument values
    char const  *ostr;      // saved options description
    char const  *place;     // option letter processing
};

} // namespace ioutil

#endif
