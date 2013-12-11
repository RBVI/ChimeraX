#include "llgr.h"
#include "llgr_int.h"

namespace llgr {

namespace internal {
AllGroups all_groups;
}
using namespace internal;

void
create_group(Id group_id)
{
	all_groups[group_id] = ObjectSet();
}

void
delete_group(Id group_id, bool and_objects)
{
	auto i = all_groups.find(group_id);
	if (i == all_groups.end())
		return;
	if (and_objects) {
		const ObjectSet &objects = i->second;
		for (auto& obj_id: objects) {
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
			const ObjectSet &objects = i.second;
			for (auto obj_id: objects) {
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
	const ObjectSet &objects = i->second;
	hide_objects(objects);
}

void
show_group(Id group_id)
{
	auto i = all_groups.find(group_id);
	if (i == all_groups.end())
		return;
	const ObjectSet &objects = i->second;
	show_objects(objects);
}

void
selection_add_group(Id group_id)
{
	auto i = all_groups.find(group_id);
	if (i == all_groups.end())
		return;
	const ObjectSet &objects = i->second;
	selection_add(objects);
}

void
selection_remove_group(Id group_id)
{
	auto i = all_groups.find(group_id);
	if (i == all_groups.end())
		return;
	const ObjectSet &objects = i->second;
	selection_remove(objects);
}

} // namespace
