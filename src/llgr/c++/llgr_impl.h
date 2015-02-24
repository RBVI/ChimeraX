// vi: set expandtab shiftwidth=4 softtabstop=4:
// Implementation of templates in llgr.h
#include "llgr_int.h"

namespace llgr {

template <> struct TypeToShaderType<int32_t *>
		{ static const ShaderType value = IVec1; };
template <> struct TypeToShaderType<int32_t [2]>
		{ static const ShaderType value = IVec2; };
template <> struct TypeToShaderType<int32_t [3]>
		{ static const ShaderType value = IVec3; };
template <> struct TypeToShaderType<int32_t [4]>
		{ static const ShaderType value = IVec4; };

template <> struct TypeToShaderType<uint32_t *>
		{ static const ShaderType value = UVec1; };
template <> struct TypeToShaderType<uint32_t [2]>
		{ static const ShaderType value = UVec2; };
template <> struct TypeToShaderType<uint32_t [3]>
		{ static const ShaderType value = UVec3; };
template <> struct TypeToShaderType<uint32_t [4]>
		{ static const ShaderType value = UVec4; };

template <> struct TypeToShaderType<float *>
		{ static const ShaderType value = FVec1; };
template <> struct TypeToShaderType<float [2]>
		{ static const ShaderType value = FVec2; };
template <> struct TypeToShaderType<float [3]>
		{ static const ShaderType value = FVec3; };
template <> struct TypeToShaderType<float [4]>
		{ static const ShaderType value = FVec4; };

template <> struct TypeToShaderType<float [2][2]>
		{ static const ShaderType value = Mat2x2; };
template <> struct TypeToShaderType<float [3][3]>
		{ static const ShaderType value = Mat3x3; };
template <> struct TypeToShaderType<float [4][4]>
		{ static const ShaderType value = Mat4x4; };

template <> struct TypeToShaderType<float [2][3]>
		{ static const ShaderType value = Mat2x3; };
template <> struct TypeToShaderType<float [3][2]>
		{ static const ShaderType value = Mat3x2; };
template <> struct TypeToShaderType<float [2][4]>
		{ static const ShaderType value = Mat2x4; };
template <> struct TypeToShaderType<float [4][2]>
		{ static const ShaderType value = Mat4x2; };
template <> struct TypeToShaderType<float [3][4]>
		{ static const ShaderType value = Mat3x4; };
template <> struct TypeToShaderType<float [4][3]>
		{ static const ShaderType value = Mat4x3; };

template <typename T> inline void
set_uniform(Id program_id, const char *name, const T &data)
{
	set_uniform(program_id, name, TypeToShaderType<T>::value, sizeof (T), data);
}

// objects

template <typename _iterable> inline void
hide_objects(const _iterable& objects)
{
	using namespace llgr::internal;
	for (auto obj_id: objects) {
		auto oii = all_objects.find(obj_id);
		if (oii == all_objects.end())
			continue;
		ObjectInfo *oi = oii->second;
		oi->hide = true;
	}
}

template <typename _iterable> inline void
show_objects(const _iterable& objects)
{
	using namespace llgr::internal;
	for (auto obj_id: objects) {
		auto oii = all_objects.find(obj_id);
		if (oii == all_objects.end())
			continue;
		ObjectInfo *oi = oii->second;
		oi->hide = false;
	}
}

template <typename _iterable> inline void
transparent(const _iterable& objects)
{
	using namespace llgr::internal;
	for (auto obj_id: objects) {
		auto oii = all_objects.find(obj_id);
		if (oii == all_objects.end())
			continue;
		ObjectInfo *oi = oii->second;
		oi->transparent = true;
	}
}

template <typename _iterable> inline void
opaque(const _iterable& objects)
{
	using namespace llgr::internal;
	for (auto obj_id: objects) {
		auto oii = all_objects.find(obj_id);
		if (oii == all_objects.end())
			continue;
		ObjectInfo *oi = oii->second;
		oi->transparent = false;
	}
}

template <typename _iterable> inline void
selection_add(const _iterable& objects)
{
	using namespace llgr::internal;
	for (auto obj_id: objects) {
		auto oii = all_objects.find(obj_id);
		if (oii == all_objects.end())
			continue;
		ObjectInfo *oi = oii->second;
		oi->selected = true;
	}
}

template <typename _iterable> inline void
selection_remove(const _iterable& objects)
{
	using namespace llgr::internal;
	for (auto obj_id: objects) {
		auto oii = all_objects.find(obj_id);
		if (oii == all_objects.end())
			continue;
		ObjectInfo *oi = oii->second;
		oi->selected = false;
	}
}

// groups

template <typename _iterable> inline void
group_add(Id group_id, const _iterable& objects)
{
	using namespace llgr::internal;
	auto i = all_groups.find(group_id);
	if (i == all_groups.end())
		return;
	ObjectSet &group_objects = i->second;
	group_objects.insert(objects.begin(), objects.end());
}

template <typename _iterable> inline void
group_remove(Id group_id, const _iterable& objects)
{
	using namespace llgr::internal;
	auto i = all_groups.find(group_id);
	if (i == all_groups.end())
		return;
	ObjectSet &group_objects = i->second;
	for (auto& obj_id: objects) {
		auto j = group_objects.find(obj_id);
		if (j == group_objects.end())
			continue;
		group_objects.erase(j);
	}
}

template <typename _iterable> inline void
render(const _iterable& groups)
{
	Groups render_groups;

	render_groups.insert(groups.begin(), groups.end());
	render(render_groups);
}

} // namespace llgr
