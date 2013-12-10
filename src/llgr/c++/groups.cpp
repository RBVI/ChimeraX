#include "llgr_int.h"

namespace llgr {

AllGroups all_groups;

void
create_group(Id group_id, const Objects& objs)
{
	all_groups[group_id] = objs;
}

void
delete_group(Id group_id, bool and_objects)
{
	auto i = all_groups.find(group_id);
	if (i == all_groups.end())
		return;
	if (and_objects) {
		const Objects &objs = i->second;
		for (auto& obj_id: objs) {
			delete_object(obj_id);
		}
		all_groups.erase(i);
	}
}

void
clear_groups(bool and_objects)
{
	if (!all_objects.empty() && and_objects) {
		// not inside clear_objects
		for (auto& i: all_groups) {
			const Objects &objs = i.second;
			for (auto obj_id: objs) {
				delete_object(obj_id);
			}
		}
	}
	all_groups.clear();
}

void
hide_group(Id group_id)
{
	auto i = all_groups.find(group_id);
	if (i == all_groups.end())
		return;
	const Objects &objs = i->second;
	hide_objects(objs);
}

void
group_add(Id group_id, Id obj_id)
{
	auto i = all_groups.find(group_id);
	if (i == all_groups.end())
		return;
	Objects &objs = i->second;
	objs.push_back(obj_id);
}

void
show_group(Id group_id)
{
	auto i = all_groups.find(group_id);
	if (i == all_groups.end())
		return;
	const Objects &objs = i->second;
	show_objects(objs);
}

void
selection_add_group(Id group_id)
{
	auto i = all_groups.find(group_id);
	if (i == all_groups.end())
		return;
	const Objects &objs = i->second;
	selection_add(objs);
}

void
selection_remove_group(Id group_id)
{
	auto i = all_groups.find(group_id);
	if (i == all_groups.end())
		return;
	const Objects &objs = i->second;
	selection_remove(objs);
}

} // namespace
