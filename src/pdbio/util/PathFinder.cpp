#include "PathFinder.h"

#include <stdlib.h>
#include <sys/types.h>

#ifdef _WIN32
# define WIN32_LEAN_AND_MEAN
# define STRICT
# define PATHSEP '\\'
# include <Windows.h>
# include <direct.h>
# include <sys/stat.h>
#else
# include <unistd.h>
# include <pwd.h>
# include <sys/stat.h>
# define PATHSEP '/'
#endif

#ifndef S_ISDIR
# define S_ISDIR(x)	(((x) & S_IFMT) == S_IFDIR)
#endif

namespace util {

bool
existsPath(const std::string &path, bool asFile)
{
#ifdef _WIN32
#ifdef _USE_32BIT_TIME_T
	struct _stat32 statbuf;
#else
	struct _stat64i32 statbuf;
#endif
	// path is UTF-8, stat needs mbcs encoding, wstat uses UTF-16
	std::vector<wchar_t> wpath(path.length() + 1);
	MultiByteToWideChar(CP_UTF8, 0, path.c_str(), path.length(),
					&wpath[0], path.length() + 1);
	if (_wstat(&wpath[0], &statbuf) < 0)
		return false;
	static const int READABLE = S_IREAD;
#else
	struct stat statbuf;
	if (stat(path.c_str(), &statbuf) < 0)
		return false;
	static const int READABLE = S_IRUSR|S_IRGRP|S_IROTH;
#endif
	bool isDir = S_ISDIR(statbuf.st_mode);
	return (statbuf.st_mode & READABLE) != 0 && (asFile ? !isDir : isDir);
}

PathFactory	*PathFactory::_path_factory = 0;

char
PathFactory::path_separator()
{
	return PATHSEP;
}

PathFactory *PathFactory::path_factory()
{
	if (_path_factory == NULL) {
		_path_factory = new PathFactory;
	}
	return _path_factory;
}

PathFinder *PathFactory::make_path_finder(const std::string &data_root,
  const std::string &package, const std::string &env,
  bool hide_data, bool use_dot_default, bool use_home_default) const
{
	return new PathFinder(data_root, package, env, hide_data,
	  use_dot_default, use_home_default);
}

PathFinder::PathFinder(const std::string &data_root,
  const std::string &package, const std::string &env, bool hide_data,
  bool use_dot_default, bool use_home_default):
  _package(package),
  _hide_data(hide_data),
  _use_dot_default(use_dot_default),
  _use_home_default(use_home_default)
{
#ifndef _WIN32
	const char *value = getenv(env.c_str());
        if (value == NULL)
#else
	std::string value;
	std::vector<wchar_t> tmpenv(env.size() + 1);
	MultiByteToWideChar(CP_UTF8, 0, &env[0], env.size(), &tmpenv[0],
							env.size() + 1);
	const wchar_t *tmp = _wgetenv(&tmpenv[0]);
	if (tmp != NULL) {
		size_t wlen = wcslen(tmp);
		int len = WideCharToMultiByte(CP_UTF8, 0, tmp, wlen,
							0, 0, 0, 0);
		value.resize(len);
		(void) WideCharToMultiByte(CP_UTF8, 0, tmp, wlen,
					&value[0], len, 0, 0);
	}
	if (value.empty())
#endif
	{
                _data_root = data_root;
        } else {
                _data_root = value;
        }
}

PathFinder::PathFinder(const PathFinder &p): _data_root(p._data_root),
	_package(p._package), _hide_data(p._hide_data),
	_use_dot_default(p._use_dot_default), _use_home_default(p._use_home_default)
{
}

// first path that has the named file/directory
// order checked is:
//	. (if 'use_dot' is true)
//	./[.]package/app (again, if 'use_dot' is true)
//	~/[.]package/app (if 'use_home' is true)
//	<data root>/app (presumably 'data root' contains package name in
//				some form)
std::string PathFinder::first_existing(const std::string &app,
	const std::string &file, bool use_dot,
	bool use_home, bool asFile, bool use_package_data) const
{
	StrList paths;
	path_list(&paths, app, file, use_package_data, use_dot, use_home);
	for (StrList::iterator pi = paths.begin(); pi != paths.end(); pi++) {
		if (existsPath(*pi, asFile))
			return *pi;
	}
	return std::string("");
}
std::string PathFinder::first_existing_file(const std::string &app,
	const std::string &file, bool use_package_data, bool use_dot,
	bool use_home) const
{
	return first_existing(app, file, use_dot, use_home, true, use_package_data);
}
std::string PathFinder::first_existing_file(const std::string &app,
		const std::string &file, bool use_dot, bool use_home) const
{
	return first_existing(app, file, use_dot, use_home, true);
}
std::string PathFinder::first_existing_dir(const std::string &app,
	const std::string &file, bool use_package_data, bool use_dot,
	bool use_home) const
{
	return first_existing(app, file, use_dot, use_home, false, use_package_data);
}
std::string PathFinder::first_existing_dir(const std::string &app,
		const std::string &file, bool use_dot, bool use_home) const
{
	return first_existing(app, file, use_dot, use_home, false);
}

// all paths that have the named file
void
PathFinder::all_existing(StrList *validPaths, const std::string &app,
	const std::string &file, bool use_dot, bool use_home, bool asFile,
	bool use_package_data) const
{
	StrList paths;
	path_list(&paths, app, file, use_package_data, use_dot, use_home);
	for (StrList::iterator pi = paths.begin(); pi != paths.end(); pi++) {
		if (existsPath(*pi, asFile))
			validPaths->push_back(*pi);
	}
}
void
PathFinder::all_existing_files(StrList *paths, const std::string &app,
	const std::string &file, bool use_package_data, bool use_dot,
	bool use_home) const
{
	all_existing(paths, app, file, use_dot, use_home, true, use_package_data);
}
void
PathFinder::all_existing_files(StrList *paths, const std::string &app,
		const std::string &file, bool use_dot, bool use_home) const
{
	all_existing(paths, app, file, use_dot, use_home, true);
}
void
PathFinder::all_existing_dirs(StrList *paths, const std::string &app,
	const std::string &file, bool use_package_data, bool use_dot,
	bool use_home) const
{
	all_existing(paths, app, file, use_dot, use_home, false, use_package_data);
}
void
PathFinder::all_existing_dirs(StrList *paths, const std::string &app,
		const std::string &file, bool use_dot, bool use_home) const
{
	all_existing(paths, app, file, use_dot, use_home, false);
}

void
PathFinder::path_list(StrList *paths, const std::string &app,
		const std::string &file, bool use_package_data, bool use_dot,
		bool use_home) const
{
	paths->reserve(4);
	std::string hidePackage = _hide_data ? "." + _package : _package;

	if (use_dot) {
		paths->push_back((std::string)"." + PATHSEP + file);
		if (app != "")
			paths->push_back((std::string)"." + PATHSEP
				+ hidePackage + PATHSEP + app + PATHSEP
				+ file);
		else
			paths->push_back((std::string)"." + PATHSEP
				+ hidePackage + PATHSEP + file);
	}

	if (use_home) {
#ifndef _WIN32
		const char *home = getenv("HOME");
#else
		const char *home = NULL;
		const wchar_t *whome = _wgetenv(L"HOME");
		if (whome == NULL)
			whome = _wgetenv(L"APPDATA");
		if (whome != NULL) {
			size_t wlen = wcslen(whome);
			int len = WideCharToMultiByte(CP_UTF8, 0, whome, wlen,
								0, 0, 0, 0);
			static std::string utfhome;
			utfhome.resize(len);
			(void) WideCharToMultiByte(CP_UTF8, 0, whome, wlen,
						&utfhome[0], len, 0, 0);

			home = utfhome.c_str();
		} else {
			// presumably Win98
			throw std::runtime_error("Windows 98 is not supported.\n");
		}
#endif
		if (home != NULL) {
			if (app != "")
				paths->push_back(std::string(home) + PATHSEP
					+ hidePackage + PATHSEP + app
					+ PATHSEP + file);
			else
				paths->push_back(std::string(home) + PATHSEP
					+ hidePackage + PATHSEP + file);
#ifndef _WIN32
		} else {
			struct passwd *pw = getpwuid(getuid());
			if (pw != NULL) {
				if (app != "")
					paths->push_back((std::string)pw->pw_dir
						+ PATHSEP + hidePackage
						+ PATHSEP + app
						+ PATHSEP + file);
				else
					paths->push_back((std::string)pw->pw_dir
						+ PATHSEP + hidePackage
						+ PATHSEP + file);
			}
#endif
		}
	}

	if (use_package_data) {
		if (app != "")
			paths->push_back(_data_root + PATHSEP + app
				+ PATHSEP + file);
		else
			paths->push_back(_data_root + PATHSEP + file);
	}
}

InputFile::InputFile(const std::string &filename): ifs_(NULL)
{
#ifndef _WIN32
	ifs_ = new std::ifstream(filename.c_str());
#else
	// filename is UTF-8, want UTF-16
	std::vector<wchar_t> wpath(filename.length() + 1);
	MultiByteToWideChar(CP_UTF8, 0, filename.c_str(), filename.length(),
					&wpath[0], filename.length() + 1);
	ifs_ = new std::ifstream(&wpath[0]);
#endif
}

InputFile::~InputFile()
{
	if (ifs_ == NULL)
		return;
	ifs_->close();
	delete ifs_;
}

std::ifstream &InputFile::ifstream() const
{
	return *ifs_;
}

OutputFile::OutputFile(const std::string &filename): ofs_(NULL)
{
#ifndef _WIN32
	ofs_ = new std::ofstream(filename.c_str());
#else
	// filename is UTF-8, want UTF-16
	std::vector<wchar_t> wpath(filename.length() + 1);
	MultiByteToWideChar(CP_UTF8, 0, filename.c_str(), filename.length(),
					&wpath[0], filename.length() + 1);
	ofs_ = new std::ofstream(&wpath[0]);
#endif
}

OutputFile::~OutputFile()
{
	if (ofs_ == NULL)
		return;
	ofs_->close();
	delete ofs_;
}

std::ofstream &OutputFile::ofstream() const
{
	return *ofs_;
}

PathFactory::PathFactory()
{
}

void
PathFinder::path_list(StrList *paths, const std::string &app, const std::string &file) const
{
	path_list(paths, app, file, true, _use_dot_default, _use_home_default);
}

void
PathFinder::path_list(StrList *paths, const std::string &app,
		const std::string &file, bool use_package_data) const
{
	path_list(paths, app, file, use_package_data, _use_dot_default,
			_use_home_default);
}

std::string
PathFinder::first_existing_file(const std::string &app,
		const std::string &file) const
{
	return first_existing_file(app, file, _use_dot_default, _use_home_default);
}

std::string
PathFinder::first_existing_dir(const std::string &app,
		const std::string &file) const
{
	return first_existing_dir(app, file, _use_dot_default, _use_home_default);
}

void
PathFinder::all_existing_files(StrList *paths, const std::string &app,
		const std::string &file) const
{
	all_existing_files(paths, app, file, _use_dot_default, _use_home_default);
}

void
PathFinder::all_existing_dirs(StrList *paths, const std::string &app,
		const std::string &file) const
{
	all_existing_dirs(paths, app, file, _use_dot_default, _use_home_default);
}

const std::string &
PathFinder::data_root() const
{
	return _data_root;
} 

bool
PathFinder::hide_data() const
{
	return _hide_data;
}

bool
PathFinder::package_dot_default() const
{
	return _use_dot_default;
}

bool
PathFinder::package_home_default() const
{
	return _use_home_default;
}

}  // namespace util
