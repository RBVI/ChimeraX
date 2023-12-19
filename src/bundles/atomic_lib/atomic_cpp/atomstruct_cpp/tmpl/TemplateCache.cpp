// vi: set expandtab ts=4 sw=4:

/*
 * === UCSF ChimeraX Copyright ===
 * Copyright 2022 Regents of the University of California. All rights reserved.
 * The ChimeraX application is provided pursuant to the ChimeraX license
 * agreement, which covers academic and commercial uses. For more details, see
 * <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
 *
 * This particular file is part of the ChimeraX library. You can also
 * redistribute and/or modify it under the terms of the GNU Lesser General
 * Public License version 2.1 as published by the Free Software Foundation.
 * For more details, see
 * <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
 *
 * THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER
 * EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. ADDITIONAL LIABILITY
 * LIMITATIONS ARE DESCRIBED IN THE GNU LESSER GENERAL PUBLIC LICENSE
 * VERSION 2.1
 *
 * This notice must be embedded in or attached to all copies, including partial
 * copies, of the software or any revisions or derivations thereof.
 * === UCSF ChimeraX Copyright ===
 */

#define ATOMSTRUCT_EXPORT
#include "TAexcept.h"
#include "TemplateCache.h"
#include <string>
#include <chutil/IOFile.h>        // use InputFile, path_exists
#include <ioutil/direntry.h>    // use ioutil::opendir(), readdir(), DIR
#include <ioutil/tokenize.h>    // use ioutil::tokenize()
#include <chutil/cmp_nocase.h>
#include <iostream>
#include <sstream>
#include <string.h>
#include <sys/stat.h>
#include "../string_types.h"

namespace tmpl {

using atomstruct::AtomName;
using atomstruct::AtomType;
using atomstruct::ResName;

#ifndef S_ISDIR
# define S_ISDIR(x)    (((x) & S_IFMT) == S_IFDIR)
#endif

using namespace ioutil;

TemplateCache *TemplateCache::_instance = nullptr;
std::string TemplateCache::_template_dir;
std::string TemplateCache::_user_template_dir;

TemplateCache *
TemplateCache::template_cache()
{
    if (_instance == nullptr) {
        _instance = new TemplateCache;
    }
    return _instance;
}

TemplateCache::AtomMap *
TemplateCache::res_template(ResName res_name, const char *app,
            const char *template_dir, const char *extension)
{
    std::string key;
    key = app;
    key += '\001';
    key += template_dir;
    key += '\001';
    key += extension;
    if (cache.find(key) == cache.end()) {
        // haven't looked this kind of template type up before
        cache_template_type(key, app, template_dir, extension);
    }
    ResMap &rm = cache[key];
    ResMap::iterator rmi = rm.find(res_name);
    if (rmi == rm.end()) {
        std::ostringstream os;
        os << "No " << app << " template found for residue " << res_name;
        throw TA_NoTemplate(os.str());
    }
    return &((*rmi).second);
}

// TODO: Might instead use C++17 std::filesystem::path::concat() when we require C++17.
static std::string
path_join(std::initializer_list<std::string> path_components)
{
    std::string path;
#ifdef _WIN32
    std::string path_sep = "\\";
#else
    std::string path_sep = "/";
#endif
    for (auto comp: path_components) {
        if (path.empty()) {
            path = comp;
        } else {
            path += path_sep;
            path += comp;
        }
    }
    return path;
}

void
TemplateCache::cache_template_type(std::string &key, const char *app,
            const char *template_dir, const char *extension)
{
    ResMap res_map;
    cache[key] = res_map;

    std::vector<std::string> search_dirs = { _user_template_dir, _template_dir };
    for (auto search_dir: search_dirs) {
        std::string t_dir = path_join({ search_dir, app, template_dir });
        DIR *tmpls = opendir(t_dir.c_str());
        if (tmpls == NULL)
            continue;

        struct dirent *entry;
        while ((entry = readdir(tmpls)) != NULL) {
	    std::string full_path = path_join({ t_dir, entry->d_name });
            struct stat s;
            if (stat(full_path.c_str(), &s) != 0)
                // couldn't stat
                continue;
            
            if (S_ISDIR(s.st_mode))
                // skip subdirectories
                continue;

            char *dot = ::strrchr(entry->d_name, '.');

            if (dot == entry->d_name)
                // ignore files that start with '.'
                continue;

            if (*extension) {
                // expect a specific extension
                if (dot == NULL)
                    // none found
                    continue;
                if (chutil::cmp_nocase(dot + 1, extension) != 0)
                    continue;
            }
            
            ResName res_name = std::string(entry->d_name, dot).c_str();
            if (res_map.find(res_name) != res_map.end())
                // residue already templated;
                // since earlier search path 
                // directories have precedence,
                // ignore this one
                  continue;

            chutil::InputFile tf(full_path);
            std::ifstream &template_file(tf.ifstream());
            if (!template_file) {
                std::cerr << "Could not read"
                << " template file " << full_path
                << std::endl;
                continue;
            }
            try {
                res_map[res_name] = parse_template_file(
                        template_file, full_path);
            } catch (...) {
                closedir(tmpls);
                throw;
            }
        }
        closedir(tmpls);
    }
    cache[key] = res_map;
}


TemplateCache::AtomMap
TemplateCache::parse_template_file(std::ifstream &template_file, std::string &path)
{
    AtomMap atom_map;
    int line_num = 0;
    for (;;) {
        line_num++;
        if (template_file.peek() == '#') {
            // skip comment lines
            template_file.ignore(1000000, '\n');
            continue;
        }
        char line[256], *fields[128];
        int num_fields;
        template_file.getline(line, sizeof line);
        if (template_file.eof())
            break;
        num_fields = ioutil::tokenize(line, fields, 128);
        if (num_fields < 0) {
            std::ostringstream os;
            os << "Cannot parse line " << line_num <<
                          "in template file " << path;
            throw TA_TemplateSyntax(os.str());
        }
        if (num_fields % 2 == 1) {
            std::ostringstream os;
            os << "Wrong number of fields in file " << path <<
                        " at line number " << line_num;
            throw TA_TemplateSyntax(os.str());
        }
        AtomName at_name(fields[0]);
        AtomType normal_at_type(fields[1]);
        ConditionalTemplate *ct = nullptr;
        if (num_fields > 2) {
            ct = new ConditionalTemplate;
            for (int field_num = 2; field_num < num_fields;
            field_num += 2) {
                try {
                    ct->add_condition(fields[field_num],
                            fields[field_num+1]);
                } catch (TA_TemplateSyntax &e) {
                    std::ostringstream os;
                    os << e.what() << " at line " <<
                        line_num << " in file " << path;
                    throw TA_TemplateSyntax(os.str());
                }
            }
        }
        atom_map[at_name] = AtomMappings(normal_at_type, ct);
    }
    return atom_map;
}

TemplateCache::~TemplateCache()
{
    for (std::map<std::string, ResMap>::iterator ci = cache.begin();
    ci != cache.end(); ++ci) {
        for (ResMap::iterator rmi = (*ci).second.begin();
        rmi != (*ci).second.end(); ++rmi) {
            for (AtomMap::iterator ami = (*rmi).second.begin();
            ami != (*rmi).second.end(); ++ami) {
                // free ConditionalTemplate pointers
                delete (*ami).second.second;
            }
        }
    }
}

void
ConditionalTemplate::add_condition(const char* cond, const AtomType& type)
{
    int operator_pos = ::strlen(cond) - 1;
    if (operator_pos < 1) {
        throw TA_TemplateSyntax(
          std::string("Bad conditional '") + cond +"'");
    }
    std::string cond_atom(cond, cond + operator_pos);
    switch ((int) *(cond + operator_pos)) {
      case '.':
      case '?':
          conditions.push_back(CondInfo(std::string(cond + operator_pos),
              cond_atom, type));
        break;
      default:
          throw TA_TemplateSyntax(
          std::string("Unknown conditional operator '") + 
          (cond + operator_pos) + "'");
        break;
    }
}

}  // namespace tmpl
