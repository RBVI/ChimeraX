#include "PDB.h"

namespace pdb {

std::istream &
operator>>(std::istream &s, PDB &p)
{
	char	buf[4 * PDB::BUF_LEN];

	s.getline(buf, 4 * PDB::BUF_LEN);
	p = PDB(buf);
	return s;
}

}  // namespace pdb
