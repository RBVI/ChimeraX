// vim: set expandtab ts=4 sw=4:

#include "Manager.h"

Group*
Global_Manager::get_group(std::string& name, bool create)
{
    auto gmi = _groups.find(name);
    if (gmi != _groups.end())
        return &(*gmi).second;

    if (!create)
        return nullptr;

    _groups.emplace(name, name);
    return &(*_groups.find(name)).second;
}
