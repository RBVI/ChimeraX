#include "llgr_int.h"

namespace llgr {

AllMatrices all_matrices;

// matrix_id of zero is reserved for identity matrix
void
create_matrix(Id matrix_id, const float mat[4][4], bool renormalize)
{
	Id data_id = --internal_buffer_id;
	create_singleton(data_id, sizeof (float [4][4]), mat);
	all_matrices[matrix_id] = MatrixInfo(data_id, renormalize);
}

void
delete_matrix(Id matrix_id)
{
	AllMatrices::iterator i = all_matrices.find(matrix_id);
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
		for (AllMatrices::iterator i = all_matrices.begin(),
					e = all_matrices.end(); i != e; ++i) {
			const MatrixInfo &info = i->second;
			delete_buffer(info.data_id);
		}
	}
	all_matrices.clear();
}

} // namespace
