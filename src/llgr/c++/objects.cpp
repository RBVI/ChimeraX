#include "llgr_int.h"
#include <algorithm>

using std::string;

namespace {

typedef std::map<string, string> NameMap;
NameMap name_map;

bool name_map_initialized = false;

void
init_name_map()
{
	if (name_map.find("position") == name_map.end())
		name_map["position"] = "position";
	if (name_map.find("normal") == name_map.end())
		name_map["normal"] = "normal";
	name_map_initialized = true;
}

const string&
attribute_alias(const string& name)
{
	NameMap::const_iterator i = name_map.find(name);
	if (i != name_map.end())
		return i->second;
	return name;
}

}

namespace llgr {

AllObjects all_objects;

void
set_attribute_alias(const string& name, const string& value)
{
	if (name != value && !value.empty())
		name_map[name] = value;
	else {
		NameMap::iterator i = name_map.find(name);
		if (i != name_map.end())
			name_map.erase(i);
	}
}

void
check_attributes(Id obj_id, Id program_id, const AttributeInfos &ais)
{
	// check for missing attributes
	AllPrograms::iterator si = all_programs.find(program_id);
	if (si == all_programs.end()) {
		std::cerr << "missing program for object " << obj_id << '\n';
	} else {
		ShaderProgram *sp = si->second;
		typedef ShaderProgram::Variables Vars;
		const Vars &attrs = sp->attributes();
		for (Vars::const_iterator j = attrs.begin(); j != attrs.end(); ++j) {
			ShaderVariable *sv = *j;
			if (sv->name() == "instanceTransform")
				continue;
			AttributeInfos::const_iterator aii = std::find_if(ais.begin(), ais.end(), AI_Name(sv->name()));
			if (aii == ais.end()) {
				std::cerr << "missing attribute " << sv->name() << " in object " << obj_id << '\n';
			}
		}
	}
}

void
create_object(Id obj_id, Id program_id, Id matrix_id, const AttributeInfos& ais, PrimitiveType pt, unsigned first, unsigned count, Id ib, DataType t)
{
	if (!name_map_initialized)
		init_name_map();
	// TODO: if (ib != 0) then t in (UByte, UShort, UInt)

	delete_object(obj_id);
	ObjectInfo *oi = new ObjectInfo(program_id, matrix_id, ais, pt, first, count, ib, t);
	all_objects[obj_id] = oi;
	dirty = true;
	for (AttributeInfos::iterator aii = oi->ais.begin();
						aii != oi->ais.end(); ++aii) {
		aii->name = attribute_alias(aii->name);
	}
	check_attributes(obj_id, program_id, oi->ais);
}

void
delete_object(Id obj_id)
{
	AllObjects::iterator i = all_objects.find(obj_id);
	if (i == all_objects.end())
		return;
	ObjectInfo *oi = i->second;
	all_objects.erase(i);
	delete oi;
}

void
clear_objects()
{
	AllObjects save;

	all_objects.swap(save);
	for (AllObjects::iterator i = save.begin(); i != save.end(); ++i) {
		delete i->second;
	}
}

void
hide_objects(const Objects& objs)
{
	for (Objects::const_iterator i = objs.begin(), e = objs.end(); i != e;
									++i) {
		Id obj_id = *i;
		AllObjects::iterator oii = all_objects.find(obj_id);
		if (oii == all_objects.end())
			continue;
		ObjectInfo *oi = oii->second;
		oi->hide = true;
	}
}

void
show_objects(const Objects& objs)
{
	for (Objects::const_iterator i = objs.begin(), e = objs.end(); i != e;
									++i) {
		Id obj_id = *i;
		AllObjects::iterator oii = all_objects.find(obj_id);
		if (oii == all_objects.end())
			continue;
		ObjectInfo *oi = oii->second;
		oi->hide = false;
	}
}

void
transparent(const Objects& objs)
{
	for (Objects::const_iterator i = objs.begin(), e = objs.end(); i != e;
									++i) {
		Id obj_id = *i;
		AllObjects::iterator oii = all_objects.find(obj_id);
		if (oii == all_objects.end())
			continue;
		ObjectInfo *oi = oii->second;
		oi->transparent = true;
	}
}

void
opaque(const Objects& objs)
{
	for (Objects::const_iterator i = objs.begin(), e = objs.end(); i != e;
									++i) {
		Id obj_id = *i;
		AllObjects::iterator oii = all_objects.find(obj_id);
		if (oii == all_objects.end())
			continue;
		ObjectInfo *oi = oii->second;
		oi->transparent = false;
	}
}

void
selection_add(const Objects& objs)
{
	for (Objects::const_iterator i = objs.begin(), e = objs.end(); i != e;
									++i) {
		Id obj_id = *i;
		AllObjects::iterator oii = all_objects.find(obj_id);
		if (oii == all_objects.end())
			continue;
		ObjectInfo *oi = oii->second;
		oi->selected = true;
	}
}

void
selection_remove(const Objects& objs)
{
	for (Objects::const_iterator i = objs.begin(), e = objs.end(); i != e;
									++i) {
		Id obj_id = *i;
		AllObjects::iterator oii = all_objects.find(obj_id);
		if (oii == all_objects.end())
			continue;
		ObjectInfo *oi = oii->second;
		oi->selected = false;
	}
}

void
selection_clear()
{
	for (AllObjects::iterator oii = all_objects.begin(),
				e = all_objects.end(); oii != e; ++oii) {
		ObjectInfo *oi = oii->second;
		oi->selected = false;
	}
}

void
clear_all()
{
	clear_objects();
#ifdef TODO
	clear_textures();
#endif
	clear_buffers();
	clear_programs();
}

} // namespace
