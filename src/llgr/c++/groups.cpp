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
	AllGroups::iterator i = all_groups.find(group_id);
	if (i == all_groups.end())
		return;
	if (and_objects) {
		const Objects &objs = i->second;
		for (Objects::const_iterator j = objs.begin(); j != objs.end();
									++j) {
			Id obj_id = *j;
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
		for (AllGroups::iterator i = all_groups.begin(),
					e = all_groups.end(); i != e; ++i) {
			const Objects &objs = i->second;
			for (Objects::const_iterator j = objs.begin();
							j != objs.end(); ++j) {
				Id obj_id = *j;
				delete_object(obj_id);
			}
		}
	}
	all_groups.clear();
}

void
hide_group(Id group_id)
{
	AllGroups::iterator i = all_groups.find(group_id);
	if (i == all_groups.end())
		return;
	const Objects &objs = i->second;
	hide_objects(objs);
}

void
show_group(Id group_id)
{
	AllGroups::iterator i = all_groups.find(group_id);
	if (i == all_groups.end())
		return;
	const Objects &objs = i->second;
	show_objects(objs);
}

void
selection_add_group(Id group_id)
{
	AllGroups::iterator i = all_groups.find(group_id);
	if (i == all_groups.end())
		return;
	const Objects &objs = i->second;
	selection_add(objs);
}

void
selection_remove_group(Id group_id)
{
	AllGroups::iterator i = all_groups.find(group_id);
	if (i == all_groups.end())
		return;
	const Objects &objs = i->second;
	selection_remove(objs);
}

} // namespace
