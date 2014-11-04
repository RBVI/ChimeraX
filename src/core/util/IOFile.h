// vim: set expandtab ts=4 sw=4:
#ifndef util_IOFile
#define util_IOFile

# include <string>
# include <fstream>
# include "imex.h"

namespace util {

bool path_exists(const std::string &path, bool asFile);

class UTIL_IMEX InputFile {
	InputFile(const InputFile&);		// disable
	InputFile& operator=(const InputFile&);	// disable
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

class UTIL_IMEX OutputFile {
	OutputFile(const OutputFile&);			// disable
	OutputFile& operator=(const OutputFile&);	// disable
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

}  // namespace util

#endif  // util_IOFile
