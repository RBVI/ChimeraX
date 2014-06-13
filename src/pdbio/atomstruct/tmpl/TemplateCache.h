// vim: set expandtab ts=4 sw=4:
#ifndef templates_TemplateCache
#define templates_TemplateCache

#include <vector>
#include <map>
#include <string>
#include "../imex.h"

class CondInfo {
public:
    std::string op, operand, result;
    CondInfo(std::string o1, std::string o2, std::string res) :
                    op(o1), operand(o2), result(res) {}
};

class ConditionalTemplate {
public:
    std::vector<CondInfo> conditions;
    void add_condition(const char *cond, const char *type);
};

class ATOMSTRUCT_IMEX TemplateCache {
public:
    typedef std::pair<std::string, ConditionalTemplate *> AtomMappings;
        // <normal IDATM type, conditional IDATM types>
    typedef std::map<std::string, AtomMappings> AtomMap;
        // atom name -> AtomMappings
    typedef std::map<std::string, AtomMap> ResMap;
        // res name -> AtomMap
    AtomMap *res_template(std::string res_name, const char *app,
            const char *template_dir, const char *extension);
    virtual ~TemplateCache();
    static TemplateCache *template_cache();
protected:
    TemplateCache() {};
private:
    std::map<std::string, ResMap> cache;
        // searchpath/extension -> resMap
    void cache_template_type(std::string &key, const char *app,
            const char *template_dir, const char *extension);
    AtomMap parse_template_file(std::ifstream &, std::string &);
    static TemplateCache *_instance;
};

#endif  // templates_TemplateCache
