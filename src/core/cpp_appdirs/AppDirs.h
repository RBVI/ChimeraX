// vim: set expandtab ts=4 sw=4:
#ifndef cpp_appdirs_AppDirs
#define cpp_appdirs_AppDirs

#include <stdexcept>
#include <string>

#include "imex.h"

namespace cpp_appdirs {
    
class CPP_APPDIRS_IMEX AppDirs {
private:
    static AppDirs*  _app_dirs;
    AppDirs(std::string& user_data_dir, std::string& user_config_dir,
        std::string& user_cache_dir, std::string& site_data_dir,
        std::string& site_config_dir, std::string& user_log_dir):
        site_config_dir(site_config_dir), site_data_dir(site_data_dir),
        user_cache_dir(user_cache_dir), user_config_dir(user_config_dir),
        user_data_dir(user_data_dir), user_log_dir(user_log_dir) {}

public:
    static const AppDirs&  get() {
        if (_app_dirs == nullptr) throw std::logic_error("C++ appdirs not"
            " initialized");
        return *_app_dirs;
    }
    static void  init_app_dirs(std::string user_data_dir,
        std::string user_config_dir, std::string user_cache_dir,
        std::string site_data_dir, std::string site_config_dir,
        std::string user_log_dir);
    const std::string  site_config_dir;
    const std::string  site_data_dir;
    const std::string  user_cache_dir;
    const std::string  user_config_dir;
    const std::string  user_data_dir;
    const std::string  user_log_dir;
};

}  // namespace cpp_appdirs

#endif  // cpp_appdirs_AppDirs
