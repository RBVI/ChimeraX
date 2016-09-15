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
    AppDirs(const std::string& path_sep, const std::string& user_data_dir,
        const std::string& user_config_dir, const std::string& user_cache_dir,
        const std::string& site_data_dir, const std::string& site_config_dir,
        const std::string& user_log_dir, const std::string& app_data_dir,
        const std::string& user_cache_dir_unversioned):
        _path_sep(path_sep), site_config_dir(site_config_dir),
        site_data_dir(site_data_dir), user_cache_dir(user_cache_dir),
        user_config_dir(user_config_dir), user_data_dir(user_data_dir),
        user_log_dir(user_log_dir), app_data_dir(app_data_dir),
        user_cache_dir_unversioned(user_cache_dir_unversioned) {}
    const std::string  _path_sep;

public:
    std::string  form_path(
            std::initializer_list<std::string> path_components) const;
    static const AppDirs&  get() {
        if (_app_dirs == nullptr) throw std::logic_error("C++ appdirs not"
            " initialized");
        return *_app_dirs;
    }
    static void  init_app_dirs(const std::string& path_sep,
        const std::string& user_data_dir, const std::string& user_config_dir,
        const std::string& user_cache_dir, const std::string& site_data_dir,
        const std::string& site_config_dir, const std::string& user_log_dir,
        const std::string& app_data_dir,
        const std::string& user_cache_dir_unversioned);
    const std::string  site_config_dir;
    const std::string  site_data_dir;
    const std::string  user_cache_dir;
    const std::string  user_config_dir;
    const std::string  user_data_dir;
    const std::string  user_log_dir;
    const std::string  app_data_dir;
    const std::string  user_cache_dir_unversioned;
};

}  // namespace appdirs

#endif  // cpp_appdirs_AppDirs
