// vim: set expandtab ts=4 sw=4:
#ifndef util_PathFinder
#define util_PathFinder

# include <string>
# include <vector>
# include <fstream>
# include "imex.h"

namespace util {

class PathFinder;

class UTIL_IMEX PathFactory {
	// singleton class
	static PathFactory *_path_factory;
	PathFactory();
public:
	static PathFactory *path_factory();
	PathFinder	*make_path_finder(const std::string &data_root,
				const std::string &package,
				const std::string &env,
				bool hide_data = true,
				bool use_dot_default = true,
				bool use_home_default = true) const;
	static char	path_separator();
};

class UTIL_IMEX PathFinder {
	void operator=(const PathFinder &p);	// disable
public:
	PathFinder(const PathFinder &p);

	// in all the below functions, 'app' can be an empty string, in which
	// case no 'app' subdirectory is assumed

	typedef std::vector<std::string> StrList;
	// The paths that will be checked, in the order they are checked
	void		path_list(/*OUT*/ StrList *paths, const std::string &app,
				const std::string &file, bool use_package_data,
				bool use_dot, bool use_home) const;
	void		path_list(/*OUT*/ StrList *paths, const std::string &app,
				const std::string &file) const;
	void		path_list(/*OUT*/ StrList *paths, const std::string &app,
				const std::string &file,
				bool use_package_data) const;

	// first path that has the named file/directory
	// order checked is:
	//      . (if 'use_dot' is true)
	//      ./package/app (again, if 'use_dot' is true)
	//      ~/package/app (if 'use_home' is true)
	//      <data root>/app (presumably, 'data root' contains package's
	//				name in some form)
	std::string	first_existing_file(const std::string &app,
				const std::string &file, bool use_package_data,
				bool use_dot, bool use_home) const;
	std::string	first_existing_file(const std::string &app,
				const std::string &file, bool use_dot,
				bool use_home) const;
	std::string	first_existing_file(const std::string &app,
				const std::string &file) const;
	std::string	first_existing_dir(const std::string &app,
				const std::string &file, bool use_package_data,
				bool use_dot, bool use_home) const;
	std::string	first_existing_dir(const std::string &app,
				const std::string &file, bool use_dot,
				bool use_home) const;
	std::string	first_existing_dir(const std::string &app,
				const std::string &file) const;

	// all paths that have the named file
	void		all_existing_files(/*OUT*/ StrList *paths,
				const std::string &app, const std::string &file,
				bool use_package_data, bool use_dot, bool use_home
			) const;
	void		all_existing_files(/*OUT*/ StrList *paths,
				const std::string &app, const std::string &file,
				bool use_dot, bool use_home) const;
	void		all_existing_files(/*OUT*/ StrList *paths,
				const std::string &app, const std::string &file
			) const;
	void		all_existing_dirs(/*OUT*/ StrList *paths,
				const std::string &app, const std::string &file,
				bool use_package_data, bool use_dot, bool use_home
			) const;
	void		all_existing_dirs(/*OUT*/ StrList *paths,
				const std::string &app, const std::string &file,
				bool use_dot, bool use_home) const;
	void		all_existing_dirs(/*OUT*/ StrList *paths,
				const std::string &app, const std::string &file
			) const;
	
	// configuration information
	const std::string & data_root() const;
	bool		hide_data() const;
	bool		package_dot_default() const;
	bool		package_home_default() const;
private:
	std::string	first_existing(const std::string &app,
				const std::string &file, bool use_dot,
				bool use_home, bool asFile, bool use_package_data = true) const;
	void		all_existing(/*OUT*/ StrList *validPaths, const std::string &app,
				const std::string &file, bool use_dot,
				bool use_home, bool asFile,
				bool use_package_data = true) const;
	friend class PathFactory;
	PathFinder(const std::string &data_root, const std::string &package,
				const std::string &env, bool hide_data,
				bool use_dot_default, bool use_home_default);

	std::string	_data_root;
	std::string	_package;
	const bool	_hide_data;
	const bool	_use_dot_default;
	const bool	_use_home_default;
};

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

#endif  // util_PathFinder
