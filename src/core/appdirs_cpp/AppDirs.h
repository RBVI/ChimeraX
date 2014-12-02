// vim: set expandtab ts=4 sw=4:
#ifndef cpp_appdirs_AppDirs
#define cpp_appdirs_AppDirs

#include <initializer_list>
#include <stdexcept>
#include <string>

#include "imex.h"

namespace appdirs {
    
class APPDIRS_IMEX AppDirs {
private:
    static AppDirs*  _app_dirs;
    AppDirs(std::string& path_sep, std::string& user_data_dir,
        std::string& user_config_dir, std::string& user_cache_dir,
        std::string& site_data_dir, std::string& site_config_dir,
        std::string& user_log_dir): _path_sep(path_sep),
        site_config_dir(site_config_dir), site_data_dir(site_data_dir),
        user_cache_dir(user_cache_dir), user_config_dir(user_config_dir),
        user_data_dir(user_data_dir), user_log_dir(user_log_dir) {}
    const std::string  _path_sep;

public:
    std::string  form_path(
            std::initializer_list<std::string> path_components) const;
    static const AppDirs&  get() {
        if (_app_dirs == nullptr) throw std::logic_error("C++ appdirs not"
            " initialized");
        return *_app_dirs;
    }
    static void  init_app_dirs(std::string path_sep,
        std::string user_data_dir, std::string user_config_dir,
        std::string user_cache_dir, std::string site_data_dir,
        std::string site_config_dir, std::string user_log_dir);
    const std::string  site_config_dir;
    const std::string  site_data_dir;
    const std::string  user_cache_dir;
    const std::string  user_config_dir;
    const std::string  user_data_dir;
    const std::string  user_log_dir;
};

}  // namespace appdirs

#endif  // cpp_appdirs_AppDirs
