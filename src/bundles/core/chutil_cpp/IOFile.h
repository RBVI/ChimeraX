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

#ifndef util_IOFile
#define util_IOFile

# include <string>
# include <fstream>

namespace chutil {

bool path_exists(const std::string &path, bool asFile);

class InputFile {
    InputFile(const InputFile&);        // disable
    InputFile& operator=(const InputFile&); // disable
public:
    // a wrapper around ifstream that guarantees that the file closes
    // when the scope is closed
    std::ifstream &ifstream() const;

    // check ifstream() for success in opening file
    InputFile(const std::string &path);
    ~InputFile();
private:
    std::ifstream *ifs_;
};

class OutputFile {
    OutputFile(const OutputFile&);          // disable
    OutputFile& operator=(const OutputFile&);   // disable
public:
    // a wrapper around ofstream that guarantees that the file closes
    // when the scope is closed
    std::ofstream &ofstream() const;

    // check ofstream() for success in opening file
    OutputFile(const std::string &path);
    ~OutputFile();
private:
    std::ofstream *ofs_;
};

}  // namespace chutil

#endif  // util_IOFile
