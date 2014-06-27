#include "PDB.h"
#include <stdio.h>
#include <ctype.h>
#include <string.h>

namespace pdb {

PDB::PDB(const char *buf)
{
	parse_line(buf);
}

void PDB::parse_line(const char *buf)
{
	const char	*fmt;

	// convert pdb record to C structure

	if (input_version == 0) {
		input_version = 2;
		if (strlen(buf) >= 80)
			// PDB ID in columns 73-76
			if (isdigit(buf[72]) && isalnum(buf[73])
			&& isalnum(buf[74]) && isalnum(buf[75]))
				input_version = 1;
	}

	memset(this, 0, sizeof *this);
	r_type = get_type(buf);
	switch (r_type) {

	case ATOM1:
	case ATOM2:
	case ATOM3:
	case ATOM4:
	case ATOM5:
	case ATOM6:
	case ATOM7:
	case ATOM8:
	case ATOM9:
	case ATOM:
	case HETATM: {
		fmt = (input_version == 2 && strlen(buf) >= 80)
		    ? "%6 %5d %4s%c%4s%c%4d%c   %8f%8f%8f%6f%6f%6 %4s%2s%2s"
		    : "%6 %5d %4s%c%4s%c%4d%c   %8f%8f%8f%6f%6f";
		if (0 <= sscanf(buf, fmt,
				&atom.serial, atom.name,
				&atom.alt_loc, atom.res.name,
				&atom.res.chain_id, &atom.res.seq_num,
				&atom.res.i_code, &atom.xyz[0],
				&atom.xyz[1], &atom.xyz[2], &atom.occupancy,
				&atom.temp_factor, atom.seg_id,
				atom.element, atom.charge)) {
			if (r_type != HETATM) {
				atom.serial += 100000 * (r_type - ATOM);
				r_type = ATOM;
			}
			break;
		}
		if (isupper(buf[6])) {
			// possible weird base-36 number for large structures
			atom_serial_number = 0;
			for (int i = 6; i < 11; ++i) {
				// for some unknown reason, if the variable 'c' is
				// declared as char, the computed value is wrong!
				int c = buf[i];
				int val;
				if (isupper(c))
					val = c - 'A' + 10;
				else if (isdigit(c))
					val = c - '0';
				else
					goto unknown;
				atom_serial_number = 36 * atom_serial_number + val;
			}
		} else if (isdigit(buf[11])) {
			// serial numbers overflowing to the right?
			char altfmt[80];
			strcpy(altfmt, "%6 %6d%4s%c%4s%c%5d   %8f%8f%8f%6f%6f");
			if (input_version == 2 && strlen(buf) >= 80)
				strcat(altfmt, "%6 %4s%2s%2s");
			if (0 <= sscanf(buf, fmt,
					&atom.serial, atom.name,
					&atom.alt_loc, atom.res.name,
					&atom.res.chain_id, &atom.res.seq_num,
					&atom.xyz[0], &atom.xyz[1], &atom.xyz[2], &atom.occupancy,
					&atom.temp_factor, atom.seg_id,
					atom.element, atom.charge)) {
				if (r_type != HETATM) {
					atom.serial += 1000000 * (r_type - ATOM);
					r_type = ATOM;
				}
				break;
			}
		} else if (strncmp(&buf[6], "*****", 5) != 0)
			goto atomqr;
		// handle atom serial number overflows (and base-36 numbers)
		char new_buf[BUF_LEN];
		strncpy(new_buf, buf, BUF_LEN);
		strncpy(&new_buf[6], "00000", 5);
		if (0 <= sscanf(new_buf, fmt,
				&atom.serial, atom.name,
				&atom.alt_loc, atom.res.name,
				&atom.res.chain_id, &atom.res.seq_num,
				&atom.res.i_code, &atom.xyz[0],
				&atom.xyz[1], &atom.xyz[2], &atom.occupancy,
				&atom.temp_factor, atom.seg_id,
				atom.element, atom.charge)) {
			atom.serial = atom_serial_number++;
			break;
		}
atomqr:
		if (0 <= sscanf(buf,
				"%6 %5d %4s%c%4s%c%4d%c   %8f%8f%8f%8f%8f",
				&atomqr.serial, atomqr.name,
				&atomqr.alt_loc, atomqr.res.name,
				&atomqr.res.chain_id, &atomqr.res.seq_num,
				&atomqr.res.i_code, &atomqr.xyz[0],
				&atomqr.xyz[1], &atomqr.xyz[2],
				&atomqr.charge, &atomqr.radius)) {
			atomqr.serial = atom_serial_number++;
			r_type = ATOMQR;
			break;
		}
		goto unknown;
	}

	case ANISOU:
		if (0 > sscanf(buf,
			"%6 %5d %4s%c%4s%c%4d%c %7d%7d%7d%7d%7d%7d  %4s%2s%2s",
				&anisou.serial, anisou.name,
				&anisou.alt_loc, anisou.res.name,
				&anisou.res.chain_id, &anisou.res.seq_num,
				&anisou.res.i_code,
				&anisou.u[0], &anisou.u[1], &anisou.u[2],
				&anisou.u[3], &anisou.u[4], &anisou.u[5],
				anisou.seg_id, anisou.element, anisou.charge))
			goto unknown;
		break;

	case AUTHOR:
		fmt = (input_version == 2) ? "%8 %2d%70s" : "%8 %2d%60s";
		if (0 > sscanf(buf, fmt, &author.continuation,
				author.author_list))
			goto unknown;
		break;

	case CISPEP:
		if (0 > sscanf(buf,
				"%7 %3d %4s%c %4d%c   %4s%c %4d%c%7 %3d%7 %6f",
				&cispep.ser_num,
				cispep.pep[0].name, &cispep.pep[0].chain_id,
				&cispep.pep[0].seq_num, &cispep.pep[0].i_code,
				cispep.pep[1].name, &cispep.pep[1].chain_id,
				&cispep.pep[1].seq_num, &cispep.pep[1].i_code,
				&cispep.mod_num, &cispep.measure))
			goto unknown;
		break;

	case COMPND:
		if (0 > sscanf(buf, "%7 %3d%60s", &compnd.continuation,
				compnd.compound))
			goto unknown;
		break;

	case CONECT:
		if (0 > sscanf(buf, "%6 %5d%5d%5d%5d%5d%5d%5d%5d%5d%5d%5d",
				&conect.serial[0], &conect.serial[1],
				&conect.serial[2], &conect.serial[3],
				&conect.serial[4], &conect.serial[5],
				&conect.serial[6], &conect.serial[7],
				&conect.serial[8], &conect.serial[9],
				&conect.serial[10]))
			goto unknown;
		break;

	case CRYST1:
		if (0 > sscanf(buf, "%6 %9f%9f%9f%7f%7f%7f %11s%4d",
				&cryst1.a, &cryst1.b, &cryst1.c,
				&cryst1.alpha, &cryst1.beta, &cryst1.gamma,
				cryst1.s_group, &cryst1.z))
			goto unknown;
		break;

	case DBREF:
		if (0 > sscanf(buf,
			"%7 %4s %c %4d%c %4d%c %6s %8s %12s %5d%c %5d%c",
				dbref.id_code, &dbref.chain_id, &dbref.seq_begin,
				&dbref.insert_begin, &dbref.seq_end,
				&dbref.insert_end, dbref.database,
				dbref.db_accession, dbref.db_id_code,
				&dbref.seq_begin2, &dbref.ins_beg_pdb,
				&dbref.seq_end2, &dbref.ins_end_pdb))
			goto unknown;
		break;

	case DBREF1:
		if (0 > sscanf(buf,
				"%7 %4s %c %4d%c %4d%c %6s%15 %20s",
				dbref1.id_code, &dbref1.chain_id,
				&dbref1.seq_begin, &dbref1.insert_begin,
				&dbref1.seq_end, &dbref1.insert_end,
				dbref1.database, dbref1.db_id_code))
			goto unknown;
		break;

	case DBREF2:
		if (0 > sscanf(buf,
				"%7 %4s %c     %22s    %10d   %10d",
				dbref2.id_code, &dbref2.chain_id,
				dbref2.db_accession,
				&dbref2.seq_begin, &dbref2.seq_end))
			goto unknown;
		break;

	case END:
	case ENDMDL:
		break;

	case EXPDTA:
		if (0 > sscanf(buf, "%8 %2d%60s", &expdta.continuation,
				expdta.technique))
			goto unknown;
		break;

	case FORMUL:
		if (0 > sscanf(buf, "%8 %2d  %4s%2d%c%51s", &formul.comp_num,
				formul.het_id, &formul.continuation,
				&formul.exclude, formul.formula))
			goto unknown;
		break;

	case FTNOTE:
		fmt = (input_version == 2) ? "%7 %3s %69s" : "%7 %3s %59s";
		if (0 > sscanf(buf, fmt, &ftnote.num, ftnote.text))
			goto unknown;
		break;

	case HEADER:
		if (0 > sscanf(buf, "%10 %40s%9s   %4s", header.classification,
				header.dep_date, header.id_code))
			goto unknown;
		break;

	case HELIX: {
		fmt = (input_version == 2)
			? "%7 %3d %3s %4s%c %4d%c %4s%c %4d%c%2d%30s %5d"
			: "%7 %3d %3s %4s%c %4d%c %4s%c %4d%c%2d%30s";
		if (0 > sscanf(buf, fmt,
				&helix.ser_num, helix.helix_id,
				helix.init.name, &helix.init.chain_id,
				&helix.init.seq_num, &helix.init.i_code,
				helix.end.name, &helix.end.chain_id,
				&helix.end.seq_num, &helix.end.i_code,
				&helix.helix_class, helix.comment,
				&helix.length))
			goto unknown;
		break;
	}

	case HET:
		fmt = (input_version == 2)
			? "%7 %4s %c%4d%c  %5d     %50s"
			: "%7 %4s %c%4d%c  %5d     %40s";
		if (0 > sscanf(buf, fmt,
				het.res.name, &het.res.chain_id, &het.res.seq_num,
				&het.res.i_code, &het.num_het_atoms,
				het.text))
			goto unknown;
		break;

	case HETNAM:
		fmt = (input_version == 2) ? "%8 %2d %4s%65s" : "%8 %2d %4s%55s";
		if (0 > sscanf(buf, fmt, &hetnam.continuation, hetnam.het_id,
				hetnam.name))
			goto unknown;
		break;

	case HETSYN:
		fmt = (input_version == 2) ? "%8 %2d %4s%65s" : "%8 %2d %4s%55s";
		if (0 > sscanf(buf, fmt, &hetsyn.continuation, hetsyn.het_id,
				hetsyn.synonyms))
			goto unknown;
		break;

	case HYDBND:
		if (0 > sscanf(buf,
		"%12 %4s%c%4s%c%5d%c %4s%c %c%5d%c %4s%c%4s%c%5d%c%6d %6d",
				hydbnd.name[0], &hydbnd.alt_loc[0],
				hydbnd.res[0].name, &hydbnd.res[0].chain_id,
				&hydbnd.res[0].seq_num, &hydbnd.res[0].i_code,
				hydbnd.name[1], &hydbnd.alt_loc[1],
				&hydbnd.res[1].chain_id,
				&hydbnd.res[1].seq_num, &hydbnd.res[1].i_code,
				hydbnd.name[2], &hydbnd.alt_loc[2],
				hydbnd.res[2].name, &hydbnd.res[2].chain_id,
				&hydbnd.res[2].seq_num, &hydbnd.res[2].i_code,
				&hydbnd.sym[0], &hydbnd.sym[1]))
			goto unknown;
		break;

	case JRNL:
		if (0 > sscanf(buf, "%12 %58s", jrnl.text))
			goto unknown;
		break;

	case KEYWDS:
		fmt = (input_version == 2) ? "%8 %2d%70s" : "%8 %2d%60s";
		if (0 > sscanf(buf, fmt, &keywds.continuation,
				keywds.keywds))
			goto unknown;
		break;

	case LINK:
		if (0 <= sscanf(buf,
			"%12 %4s%c%4s%c%4d%c%15 %4s%c%4s%c%4d%c  %6d %6d %6f",
				link.name[0], &link.alt_loc[0],
				link.res[0].name, &link.res[0].chain_id,
				&link.res[0].seq_num, &link.res[0].i_code,
				link.name[1], &link.alt_loc[1],
				link.res[1].name, &link.res[1].chain_id,
				&link.res[1].seq_num, &link.res[1].i_code,
				&link.sym[0], &link.sym[1], &link.length))
			break;
		// Coot/Refmac don't supply the distance; accept anyway
		if (0 <= sscanf(buf,
			"%12 %4s%c%4s%c%4d%c%15 %4s%c%4s%c%4d%c  %6d %6d",
				link.name[0], &link.alt_loc[0],
				link.res[0].name, &link.res[0].chain_id,
				&link.res[0].seq_num, &link.res[0].i_code,
				link.name[1], &link.alt_loc[1],
				link.res[1].name, &link.res[1].chain_id,
				&link.res[1].seq_num, &link.res[1].i_code,
				&link.sym[0], &link.sym[1])) {
			link.length = -1.0;
			break;
		}
		goto unknown;

	case MASTER:
		if (0 > sscanf(buf, "%10 %5d%5d%5d%5d%5d%5d%5d%5d%5d%5d%5d%5d",
				&master.num_remark, &master.num_ftnote,
				&master.num_het, &master.num_helix,
				&master.num_sheet, &master.num_turn,
				&master.num_site, &master.num_xform,
				&master.num_coord, &master.num_ter,
				&master.num_conect, &master.num_seq))
			goto unknown;
		break;

	case MDLTYP:
		if (0 > sscanf(buf, "%8 %2d%70s", &mdltyp.continuation,
				mdltyp.comment))
			goto unknown;
		break;

	case MODEL:
		if (0 > sscanf(buf, "%6 %8d", &model.serial))
			goto unknown;
		break;
	
	case MODRES:
		fmt = (input_version == 2)
			? "%7 %4s %4s%c %4d%c %4s %51s"
			: "%7 %4s %4s%c %4d%c %4s %41s";
		if (0 > sscanf(buf, fmt,
				modres.id_code, modres.res.name,
				&modres.res.chain_id, &modres.res.seq_num,
				&modres.res.i_code, modres.std_res,
				modres.comment))
			goto unknown;
		break;

	case MTRIX:
		if (0 > sscanf(buf, "%5 %d %3d%10f%10f%10f%5 %10f    %d",
				&mtrix.row_num, &mtrix.serial, &mtrix.m[0],
				&mtrix.m[1], &mtrix.m[2], &mtrix.v,
				&mtrix.i_given))
			goto unknown;
		break;

	case NUMMDL:
		if (0 > sscanf(buf, "%10 %4d", &nummdl.model_number))
			goto unknown;
		break;

	case OBSLTE:
		if (0 > sscanf(buf,
			"%8 %2d %9s %4s%6 %4s %4s %4s %4s %4s %4s %4s %4s",
				&obslte.continuation, obslte.rep_date,
				obslte.id_code, obslte.r_id_code[0],
				obslte.r_id_code[1], obslte.r_id_code[2],
				obslte.r_id_code[3], obslte.r_id_code[4],
				obslte.r_id_code[2], obslte.r_id_code[6],
				obslte.r_id_code[7]))
			goto unknown;
		break;

	case ORIGX:
		if (0 > sscanf(buf, "%5 %d%4 %10f%10f%10f%5 %10f",
				&origx.row_num, &origx.o[0], &origx.o[1],
				&origx.o[2], &origx.t))
			goto unknown;
		break;

	case REMARK:
		fmt = (input_version == 2) ? "%7 %3d %69s" : "%7 %3d %59s";
		if (0 > sscanf(buf, fmt, &remark.remark_num, remark.empty))
			goto unknown;
		break;

	case REVDAT:
		if (0 > sscanf(buf, "%7 %3d%2d %9s %5s   %d%7 %6s %6s %6s %6s",
				&revdat.mod_num,
				&revdat.continuation, revdat.mod_date,
				revdat.mod_id, &revdat.mod_type,
				revdat.record[0], revdat.record[1],
				revdat.record[2], revdat.record[3]))
			goto unknown;
		break;

	case SCALE:
		if (0 > sscanf(buf, "%5 %d%4 %10f%10f%10f%5 %10f",
				&scale.row_num, &scale.s[0], &scale.s[1],
				&scale.s[2], &scale.u))
			goto unknown;
		break;

	case SEQADV:
		fmt = (input_version == 2)
			? "%7 %4s %4s%c %4d%c %4s %9s %4s%5d %31s"
			: "%7 %4s %4s%c %4d%c %4s %9s %4s%5d %21s";
		if (0 > sscanf(buf, fmt,
				seqadv.id_code, seqadv.res.name,
				&seqadv.res.chain_id, &seqadv.res.seq_num,
				&seqadv.res.i_code, seqadv.database,
				seqadv.db_id_code, seqadv.db_res, &seqadv.db_seq,
				seqadv.conflict))
			goto unknown;
		break;

	case SEQRES:
		if (0 > sscanf(buf,
		"%7 %3d %c %4d  %4s%4s%4s%4s%4s%4s%4s%4s%4s%4s%4s%4s%4s",
				&seqres.ser_num, &seqres.chain_id,
				&seqres.num_res, seqres.res_name[0],
				seqres.res_name[1], seqres.res_name[2],
				seqres.res_name[3], seqres.res_name[4],
				seqres.res_name[5], seqres.res_name[6],
				seqres.res_name[7], seqres.res_name[8],
				seqres.res_name[9], seqres.res_name[10],
				seqres.res_name[11], seqres.res_name[12]))
			goto unknown;
		break;

	case SHEET:
		if (0 > sscanf(buf,
	"%7 %3d %3s%2d %4s%c%4d%c %4s%c%4d%c%2d %4s%4s%c%4d%c %4s%4s%c%4d%c",
				&sheet.strand, sheet.sheet_id,
				&sheet.num_strands, sheet.init.name,
				&sheet.init.chain_id, &sheet.init.seq_num,
				&sheet.init.i_code, sheet.end.name,
				&sheet.end.chain_id, &sheet.end.seq_num,
				&sheet.end.i_code, &sheet.sense,
				sheet.cur_atom, sheet.cur.name,
				&sheet.cur.chain_id, &sheet.cur.seq_num,
				&sheet.cur.i_code, sheet.prev_atom,
				sheet.prev.name, &sheet.prev.chain_id,
				&sheet.prev.seq_num, &sheet.prev.i_code))
			goto unknown;
		break;

	case SIGATM: {
		int ftnote_num;		// backwards compatibility
		if (0 <= sscanf(buf,
			"%6 %5d %4s%c%4s%c%4d%c   %8f%8f%8f%6f%6f %3d  %4s%2s%2s",
				&sigatm.serial, sigatm.name,
				&sigatm.alt_loc, sigatm.res.name,
				&sigatm.res.chain_id, &sigatm.res.seq_num,
				&sigatm.res.i_code, &sigatm.sig_xyz[0],
				&sigatm.sig_xyz[1], &sigatm.sig_xyz[2],
				&sigatm.sig_occ, &sigatm.sig_temp, &ftnote_num,
				sigatm.seg_id, sigatm.element, sigatm.charge))
			break;
		// handle atom serial number overflows
		if (strncmp(&buf[6], "*****", 5) != 0)
			goto unknown;
		char new_buf[BUF_LEN];
		strncpy(new_buf, buf, BUF_LEN);
		strncpy(&new_buf[6], "00000", 5);
		if (0 <= sscanf(new_buf,
			"%6 %5d %4s%c%4s%c%4d%c   %8f%8f%8f%6f%6f %3d  %4s%2s%2s",
				&sigatm.serial, sigatm.name,
				&sigatm.alt_loc, sigatm.res.name,
				&sigatm.res.chain_id, &sigatm.res.seq_num,
				&sigatm.res.i_code, &sigatm.sig_xyz[0],
				&sigatm.sig_xyz[1], &sigatm.sig_xyz[2],
				&sigatm.sig_occ, &sigatm.sig_temp, &ftnote_num,
				sigatm.seg_id, sigatm.element, sigatm.charge)) {
			sigatm.serial = sigatm_serial_number++;
			break;
		}
		goto unknown;
	}

	case SIGUIJ:
		if (0 > sscanf(buf,
			"%6 %5d %4s%c%4s%c%4d%c %7d%7d%7d%7d%7d%7d  %4s%2s%2s",
				&siguij.serial, siguij.name,
				&siguij.alt_loc, siguij.res.name,
				&siguij.res.chain_id,
				&siguij.res.seq_num,
				&siguij.res.i_code,
				&siguij.sig[0], &siguij.sig[1], &siguij.sig[2],
				&siguij.sig[3], &siguij.sig[4], &siguij.sig[5],
				siguij.seg_id, siguij.element, siguij.charge))
			goto unknown;
		break;

	case SITE:
		if (0 > sscanf(buf,
		"%7 %3d %3s %2d %4s%c%4d%c %4s%c%4d%c %4s%c%4d%c %4s%c%4d%c",
				&site.seq_num, site.site_id, &site.num_res,
				site.res[0].name, &site.res[0].chain_id,
				&site.res[0].seq_num, &site.res[0].i_code,
				site.res[1].name, &site.res[1].chain_id,
				&site.res[1].seq_num, &site.res[1].i_code,
				site.res[2].name, &site.res[2].chain_id,
				&site.res[2].seq_num, &site.res[2].i_code,
				site.res[3].name, &site.res[3].chain_id,
				&site.res[3].seq_num, &site.res[3].i_code))
			goto unknown;
		break;

	case SLTBRG:
		if (0 > sscanf(buf,
			"%12 %4s%c%4s%c%4d%c%15 %4s%c%4s%c%4d%c  %6d %6d",
				sltbrg.name[0], &sltbrg.alt_loc[0],
				sltbrg.res[0].name, &sltbrg.res[0].chain_id,
				&sltbrg.res[0].seq_num, &sltbrg.res[0].i_code,
				sltbrg.name[1], &sltbrg.alt_loc[1],
				sltbrg.res[1].name, &sltbrg.res[1].chain_id,
				&sltbrg.res[1].seq_num, &sltbrg.res[1].i_code,
				&sltbrg.sym[0], &sltbrg.sym[1]))
			goto unknown;
		break;

	case SOURCE:
		if (0 > sscanf(buf, "%7 %3d%60s", &source.continuation,
				source.src_name))
			goto unknown;
		break;
	
	case CAVEAT:
		if (0 > sscanf(buf, "%8 %2d %4s    %61s", &caveat.continuation,
				caveat.id_code, caveat.comment))
			goto unknown;
		break;

	case SPLIT:
		if (0 > sscanf(buf,
	"%8 %2d %4s %4s %4s %4s %4s %4s %4s %4s %4s %4s %4s %4s %4s %4s",
				&split.continuation,
				split.id_code[0], split.id_code[1],
				split.id_code[2], split.id_code[3],
				split.id_code[4], split.id_code[5],
				split.id_code[6], split.id_code[7],
				split.id_code[8], split.id_code[9],
				split.id_code[10], split.id_code[11],
				split.id_code[12], split.id_code[13]))
			goto unknown;
		break;

	case SPRSDE:
		if (0 > sscanf(buf,
			"%8 %2d %9s %4s%6 %4s %4s %4s %4s %4s %4s %4s %4s",
				&sprsde.continuation,
				sprsde.sprsde_date, sprsde.id_code,
				sprsde.s_id_code[0], sprsde.s_id_code[1],
				sprsde.s_id_code[2], sprsde.s_id_code[3],
				sprsde.s_id_code[4], sprsde.s_id_code[5],
				sprsde.s_id_code[6], sprsde.s_id_code[7]))
			goto unknown;
		break;

	case SSBOND:
		if (0 > sscanf(buf,
			"%7 %3d %4s%c %4d%c   %4s%c %4d%c%23 %6d %6d %5f",
				&ssbond.ser_num,
				ssbond.res[0].name, &ssbond.res[0].chain_id,
				&ssbond.res[0].seq_num, &ssbond.res[0].i_code,
				ssbond.res[1].name, &ssbond.res[1].chain_id,
				&ssbond.res[1].seq_num, &ssbond.res[1].i_code,
				&ssbond.sym[0], &ssbond.sym[1], &ssbond.length))
			goto unknown;
		break;

	case TER:
		if (0 > sscanf(buf, "%6 %5d%6 %4s%c%4d%c", &ter.serial,
				ter.res.name, &ter.res.chain_id, &ter.res.seq_num,
				&ter.res.i_code))
			goto unknown;
		break;
	
	case TITLE:
		if (0 > sscanf(buf, "%8 %2d%60s", &title.continuation,
				title.title))
			goto unknown;
		break;

	case TURN:
		if (0 > sscanf(buf, "%7 %3d %3s %4s%c%4d%c %4s%c%4d%c    %30s",
				&turn.seq, turn.turn_id,
				turn.init.name, &turn.init.chain_id,
				&turn.init.seq_num, &turn.init.i_code,
				turn.end.name, &turn.end.chain_id,
				&turn.end.seq_num, &turn.end.i_code,
				turn.comment))
			goto unknown;
		break;

	case TVECT:
		if (0 > sscanf(buf, "%7 %3d%10f%10f%10f%30s", &tvect.serial,
				&tvect.t[0], &tvect.t[1], &tvect.t[2],
				tvect.comment))
			goto unknown;
		break;

user:
		r_type = USER;
	case USER:
		if (0 > sscanf(buf, "%4 %2s%74s", user.subtype, user.text))
			goto unknown;
		break;

	default:
	case UNKNOWN:
unknown:
		r_type = UNKNOWN;		// in case of goto
		::sprintf(unknown.junk, "%.*s", BUF_LEN, buf);
		break;
	}
}

}  // namespace pdb
