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

#include <stdexcept>

#define APPDIRS_EXPORT
#include "AppDirs.h"

namespace appdirs {

using std::string;

AppDirs*  AppDirs::_app_dirs = nullptr;

string
AppDirs::form_path(std::initializer_list<string> path_components) const
{
    string path;
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
AppDirs::init_app_dirs(const string& path_sep,
    const string& user_data_dir, const string& user_config_dir,
    const string& user_cache_dir, const string& site_data_dir,
    const string& site_config_dir, const string& user_log_dir,
    const string& app_data_dir, const string& user_cache_dir_unversioned)
{
    if (_app_dirs != nullptr)
        throw std::logic_error("C++ appdirs already initialized!");
    _app_dirs = new AppDirs(path_sep, user_data_dir, user_config_dir,
        user_cache_dir, site_data_dir, site_config_dir, user_log_dir,
        app_data_dir, user_cache_dir_unversioned);
}

}; // namespace appdirs
