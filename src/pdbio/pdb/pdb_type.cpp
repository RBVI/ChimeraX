#include "PDB.h"
#include <ctype.h>
#include <string.h>

#ifdef _WIN32
# define strcasecmp	strcmp
# define strncasecmp	strncmp
#else
// strcasecmp and strncasecmp are not ISO C, but available on most platforms
extern "C" int strcasecmp (const char *s1, const char *s2);
extern "C" int strncasecmp (const char *s1, const char *s2, size_t n);
#endif

# ifndef _toupper
# define	_toupper	toupper
# endif

namespace pdb {

int PDB::input_version = 0;
int PDB::atom_serial_number = 10000;
int PDB::sigatm_serial_number = 10000;

PDB::RecordType
PDB::get_type(const char *buf)
{
	char	rt[7];		// PDB record type
	int	i;

	for (i = 0; buf[i] != '\0' && buf[i] != '\n' && i < 6; i += 1) {
		if (islower(buf[i]))
			rt[i] = _toupper(buf[i]);
		else
			rt[i] = buf[i];
	}
	if (i < 6)
		for (; i < 6; i += 1)
			rt[i] = ' ';
	rt[6] = '\0';

	switch (rt[0]) {

	case 'A':
		switch (rt[1]) {
		case 'N':
			if (strcmp(rt + 2, "ISOU") == 0)
				return ANISOU;
			break;
		case 'T':
			if (strcmp(rt + 2, "OM  ") == 0)
				return ATOM;
			if (rt[4] == ' ' && rt[5] >= '1' && rt[5] <= '9')
				return (PDB::RecordType)(ATOM + (rt[5] - '0'));
			break;
		case 'U':
			if (strcmp(rt + 2, "THOR") == 0)
				return AUTHOR;
			break;
		}
		break;

	case 'C':
		switch (rt[1]) {
		case 'A':
			if (strcmp(rt + 2, "VEAT") == 0)
				return CAVEAT;
			break;
		case 'I':
			if (strcmp(rt + 2, "SPEP") == 0)
				return CISPEP;
			break;
		case 'O':
			if (strcmp(rt + 2, "MPND") == 0)
				return COMPND;
			if (strcmp(rt + 2, "NECT") == 0)
				return CONECT;
			break;
		case 'R':
			if (strcmp(rt + 2, "YST1") == 0)
				return CRYST1;
			break;
		}
		break;

	case 'D':
		if (strcmp(rt + 1, "BREF ") == 0)
			return DBREF;
		if (strcmp(rt + 1, "BREF1") == 0)
			return DBREF1;
		if (strcmp(rt + 1, "BREF2") == 0)
			return DBREF2;
		break;

	case 'E':
		switch (rt[1]) {
		case 'N':
			if (strcmp(rt + 2, "D   ") == 0)
				return END;
			if (strcmp(rt + 2, "DMDL") == 0)
				return ENDMDL;
			break;
		case 'X':
			if (strcmp(rt + 2, "PDTA") == 0)
				return EXPDTA;
			break;
		}
		break;

	case 'F':
		switch (rt[1]) {
		case 'T':
			if (strcmp(rt + 2, "NOTE") == 0)
				return FTNOTE;
			break;
		case 'O':
			if (strcmp(rt + 2, "RMUL") == 0)
				return FORMUL;
			break;
		}
		break;

	case 'H':
		switch (rt[1]) {
		case 'E':
			if (strcmp(rt + 2, "TATM") == 0)
				return HETATM;
			if (strcmp(rt + 2, "ADER") == 0)
				return HEADER;
			if (strcmp(rt + 2, "T   ") == 0)
				return HET;
			if (strcmp(rt + 2, "TNAM") == 0)
				return HETNAM;
			if (strcmp(rt + 2, "TSYN") == 0)
				return HETSYN;
			if (strcmp(rt + 2, "LIX ") == 0)
				return HELIX;
			break;
		case 'Y':
			if (strcmp(rt + 2, "DBND") == 0)
				return HYDBND;
			break;
		}
		break;

	case 'J':
		if (strcmp(rt + 1, "RNL  ") == 0)
			return JRNL;
		break;

	case 'K':
		if (strcmp(rt + 1, "EYWDS") == 0)
			return KEYWDS;
		break;

	case 'L':
		if (strcmp(rt + 1, "INK  ") == 0)
			return LINK;
		break;

	case 'M':
		switch (rt[1]) {
		case 'A':
			if (strcmp(rt + 2, "STER") == 0)
				return MASTER;
			break;
		case 'D':
			if (strcmp(rt + 2, "LTYP") == 0)
				return MDLTYP;
			break;
		case 'O':
			if (strcmp(rt + 2, "DEL ") == 0)
				return MODEL;
			if (strcmp(rt + 2, "DRES") == 0)
				return MODRES;
			break;
		case 'T':
			if (strcmp(rt + 2, "RIX1") == 0
			|| strcmp(rt + 2, "RIX2") == 0
			|| strcmp(rt + 2, "RIX3") == 0)
				return MTRIX;
			break;
		}
		break;

	case 'N':
		switch (rt[1]) {
		case 'U':
			if (strcmp(rt + 2, "MMDL") == 0)
				return NUMMDL;
			break;
		}
		break;

	case 'O':
		switch (rt[1]) {
		case 'B':
			if (strcmp(rt + 2, "SLTE") == 0)
				return OBSLTE;
			break;
		case 'R':
			if (strcmp(rt + 2, "IGX1") == 0
			|| strcmp(rt + 2, "IGX2") == 0
			|| strcmp(rt + 2, "IGX3") == 0)
				return ORIGX;
			break;
		}
		break;

	case 'R':
		if (rt[1] != 'E')
			break;
		if (strcmp(rt + 2, "MARK") == 0)
			return REMARK;
		if (strcmp(rt + 2, "VDAT") == 0)
			return REVDAT;
		break;

	case 'S':
		switch (rt[1]) {
		case 'C':
			if (strcmp(rt + 2, "ALE1") == 0
			|| strcmp(rt + 2, "ALE2") == 0
			|| strcmp(rt + 2, "ALE3") == 0)
				return SCALE;
			break;
		case 'E':
			if (strcmp(rt + 2, "QRES") == 0)
				return SEQRES;
			if (strcmp(rt + 2, "QADV") == 0)
				return SEQADV;
			break;
		case 'H':
			if (strcmp(rt + 2, "EET ") == 0)
				return SHEET;
			break;
		case 'I':
			if (strcmp(rt + 2, "TE  ") == 0)
				return SITE;
			if (strcmp(rt + 2, "GATM") == 0)
				return SIGATM;
			if (strcmp(rt + 2, "GUIJ") == 0)
				return SIGUIJ;
			break;
		case 'L':
			if (strcmp(rt + 2, "TBRG") == 0)
				return SLTBRG;
			break;
		case 'O':
			if (strcmp(rt + 2, "URCE") == 0)
				return SOURCE;
			break;
		case 'P':
			if (strcmp(rt + 2, "RSDE") == 0)
				return SPRSDE;
			if (strcmp(rt + 2, "LIT ") == 0)
				return SPLIT;
			break;
		case 'S':
			if (strcmp(rt + 2, "BOND") == 0)
				return SSBOND;
			break;
		}
		break;

	case 'T':
		switch (rt[1]) {
		case 'E':
			if (strcmp(rt + 2, "R   ") == 0)
				return TER;
			break;
		case 'I':
			if (strcmp(rt + 2, "TLE ") == 0)
				return TITLE;
			break;
		case 'U':
			if (strcmp(rt + 2, "RN  ") == 0)
				return TURN;
			break;
		case 'V':
			if (strcmp(rt + 2, "ECT ") == 0)
				return TVECT;
			break;
		}
		break;

	case 'U':
		if (rt[1] == 'S' && rt[2] == 'E' && rt[3] == 'R')
			return USER;
		break;
	}
	return UNKNOWN;
}

}  // namespace pdb
