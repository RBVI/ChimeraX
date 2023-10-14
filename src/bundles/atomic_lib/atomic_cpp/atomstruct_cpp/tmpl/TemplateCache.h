// vi: set expandtab ts=4 sw=4:

/*
 * === UCSF ChimeraX Copyright ===
 * Copyright 2022 Regents of the University of California. All rights reserved.
 * This software is provided pursuant to the ChimeraX license agreement, which
 * covers academic and commercial uses. For more information, see
 * <http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html>
 *
 * This file is part of the ChimeraX library. You can also redistribute and/or
 * modify it under the GNU Lesser General Public License version 2.1 as
 * published by the Free Software Foundation. For more details, see
 * <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html>
 *
 * This file is distributed WITHOUT ANY WARRANTY; without even the implied
 * warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This notice
 * must be embedded in or attached to all copies, including partial copies, of
 * the software or any revisions or derivations thereof.
 * === UCSF ChimeraX Copyright ===
 */

#ifndef templates_TemplateCache
#define templates_TemplateCache

#include <vector>
#include <map>
#include <string>
#include "../imex.h"
#include "../string_types.h"

namespace tmpl {

using atomstruct::AtomName;
using atomstruct::AtomType;
using atomstruct::ResName;
    
class ATOMSTRUCT_IMEX CondInfo {
public:
    std::string op, operand;
    AtomType result;
    CondInfo(std::string o1, std::string o2, AtomType res) :
                    op(o1), operand(o2), result(res) {}
};

class ATOMSTRUCT_IMEX ConditionalTemplate {
public:
    std::vector<CondInfo> conditions;
    void add_condition(const char *cond, const AtomType& type);
};

class ATOMSTRUCT_IMEX TemplateCache {
public:
    typedef std::pair<AtomType, ConditionalTemplate *> AtomMappings;
        // <normal IDATM type, conditional IDATM types>
    typedef std::map<AtomName, AtomMappings> AtomMap;
        // atom name -> AtomMappings
    typedef std::map<ResName, AtomMap> ResMap;
        // res name -> AtomMap
    AtomMap *res_template(ResName res_name, const char *app,
            const char *template_dir, const char *extension);
    static void set_templates_dir(const std::string& template_dir) { _template_dir = template_dir; }
    static void set_user_templates_dir(const std::string& template_dir) { _user_template_dir = template_dir; }
    virtual ~TemplateCache();
    static TemplateCache *template_cache();
protected:
    TemplateCache() {};
private:
    static std::string _template_dir;
    static std::string _user_template_dir;
    std::map<std::string, ResMap> cache;
        // searchpath/extension -> resMap
    void cache_template_type(std::string &key, const char *app,
            const char *template_dir, const char *extension);
    AtomMap parse_template_file(std::ifstream &, std::string &);
    static TemplateCache *_instance;
};

#endif  // templates_TemplateCache

}  // namespace tmpl
