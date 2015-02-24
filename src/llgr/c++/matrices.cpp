// vi: set expandtab shiftwidth=4 softtabstop=4:
#include "llgr.h"
#include "llgr_int.h"

namespace llgr {

namespace internal {
AllMatrices all_matrices;
}
using namespace internal;

// matrix_id of zero is reserved for identity matrix
void
create_matrix(Id matrix_id, const float mat[4][4], bool renormalize)
{
	Id data_id;
	auto i = all_matrices.find(matrix_id);
	if (i == all_matrices.end())
		data_id = --internal_buffer_id;
	else {
		const MatrixInfo &info = i->second;
		data_id = info.data_id;
	}
	create_singleton(data_id, sizeof (float [4][4]), mat);
	all_matrices[matrix_id] = MatrixInfo(data_id, renormalize);
}

void
delete_matrix(Id matrix_id)
{
	auto i = all_matrices.find(matrix_id);
	if (i != all_matrices.end()) {
		const MatrixInfo &info = i->second;
		delete_buffer(info.data_id);
		all_matrices.erase(i);
	}
}

void
clear_matrices()
{
	if (!all_buffers.empty()) {
		// not inside clear_buffers
		for (auto& i: all_matrices) {
			const MatrixInfo &info = i.second;
			delete_buffer(info.data_id);
		}
	}
	all_matrices.clear();
}

} // namespace
