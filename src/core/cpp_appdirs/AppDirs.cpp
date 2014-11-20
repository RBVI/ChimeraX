#include <stdexcept>

#include "AppDirs.h"

namespace appdirs {

AppDirs*  AppDirs::_app_dirs = nullptr;

std::string
AppDirs::form_path(std::initializer_list<std::string> path_components) const
{
	std::string path;
	for (auto comp: path_components) {
		if (path.empty()) {
			path = comp;
		} else {
			path += _path_sep;
			path += comp;
		}
	}
	return path;
}
	
void
AppDirs::init_app_dirs(std::string path_sep,
	std::string user_data_dir, std::string user_config_dir,
	std::string user_cache_dir, std::string site_data_dir,
	std::string site_config_dir, std::string user_log_dir)
{
	if (_app_dirs != nullptr)
		throw std::logic_error("C++ appdirs already initialized!");
	_app_dirs = new AppDirs(path_sep, user_data_dir, user_config_dir,
		user_cache_dir, site_data_dir, site_config_dir, user_log_dir);
}

}; // namespace cpp_appdirs
