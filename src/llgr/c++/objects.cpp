#include "llgr.h"
#include "llgr_int.h"
#include <algorithm>

using std::string;

namespace {

typedef std::unordered_map<string, string> NameMap;
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

}

namespace llgr {

namespace internal {
AllObjects all_objects;
}
using namespace internal;

const string&
attribute_alias(const string& name)
{
	if (!name_map_initialized)
		init_name_map();
	auto i = name_map.find(name);
	if (i != name_map.end())
		return i->second;
	return name;
}

void
set_attribute_alias(const string& name, const string& value)
{
	if (name != value && !value.empty())
		name_map[name] = value;
	else {
		auto i = name_map.find(name);
		if (i != name_map.end())
			name_map.erase(i);
	}
}

void
check_attributes(Id obj_id, ObjectInfo *oi)
{
	// check for missing attributes and setup vertex arrays
	auto si = all_programs.find(oi->program_id);
	if (si == all_programs.end()) {
		std::cerr << "missing program for object " << obj_id << '\n';
		return;
	}

#ifdef USE_VAO
	oi->singleton_cache.clear();
	oi->singleton_cache.reserve(2);
#endif
	ShaderProgram *sp = si->second;
	for (const auto sv: sp->attributes()) {
		if (sv->name() == "instanceTransform")
			continue;
		auto aii = std::find_if(oi->ais.begin(), oi->ais.end(), AI_Name(sv->name()));
		if (aii == oi->ais.end()) {
			std::cerr << "missing attribute " << sv->name() << " in object " << obj_id << '\n';
		}
#ifdef USE_VAO
		const AttributeInfo &ai = *aii;
		auto bii = all_buffers.find(ai.data_id);
		if (bii == all_buffers.end())
			return;
		const BufferInfo &bi = bii->second;
		int loc = sv->location();
		unsigned num_locations, num_elements;
		attr_location_info(sv->type(), &num_locations, &num_elements);
		if (bi.data) {
			oi->singleton_cache.push_back(SingletonInfo(ai.type,
				ai.normalized, bi.data, loc, num_locations,
				num_elements));
		} else {
			setup_array_attribute(bi, ai, loc, num_locations);
		}
#endif
	}
#ifdef USE_VAO
	if (oi->index_buffer_id) {
		auto bii = all_buffers.find(oi->index_buffer_id);
		if (bii != all_buffers.end()) {
			const BufferInfo &ibi = bii->second;
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibi.buffer);
		}
	}
	oi->cache_valid = true;
#endif
}

void
create_object(Id obj_id, Id program_id, Id matrix_id, const AttributeInfos& ais, PrimitiveType pt, unsigned first, unsigned count, Id ib, DataType t)
{
	if (!name_map_initialized)
		init_name_map();
	
	if (ib && t != UByte && t != UShort && t != UInt)
		throw std::logic_error("DataType must be an unsigned type");

	ObjectInfo *oi = new ObjectInfo(program_id, matrix_id, ais, pt, first, count, ib, t);
	delete_object(obj_id);
	all_objects[obj_id] = oi;
	dirty = true;
	for (auto ai: oi->ais) {
		ai.name = attribute_alias(ai.name);
	}
#ifdef USE_VAO
	glGenVertexArrays(1, &oi->vao);
	glBindVertexArray(oi->vao);
#endif
	check_attributes(obj_id, oi);
#ifdef USE_VAO
	glBindVertexArray(0);
#endif
}

void
delete_object(Id obj_id)
{
	auto i = all_objects.find(obj_id);
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
	for (auto i: save) {
		delete i.second;
	}
	clear_groups();
}

void
selection_clear()
{
	for (auto oii: all_objects) {
		ObjectInfo *oi = oii.second;
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
