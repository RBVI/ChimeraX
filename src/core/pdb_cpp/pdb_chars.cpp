// vi: set expandtab ts=4 sw=4:
#include "PDB.h"
#include <stdio.h>
#include <ctype.h>

namespace pdb {

const char *
PDB::c_str(void) const
{
    static char buf[BUF_LEN];
    const char  *fmt;
    int     count;

    // convert C structure to pdb record

    switch (r_type) {

    case UNKNOWN:
        count = ::sprintf(buf, "%.*s", BUF_LEN - 2, unknown.junk);
        break;

    case ANISOU:
        count = sprintf(buf,
        "ANISOU%5d %-4s%c%-4s%c%4d%c %7d%7d%7d%7d%7d%7d  %-4s%2s%-.2s",
            anisou.serial, anisou.name, anisou.alt_loc,
            anisou.res.name, anisou.res.chain_id, anisou.res.seq_num,
            anisou.res.i_code, anisou.u[0], anisou.u[1],
            anisou.u[2], anisou.u[3], anisou.u[4], anisou.u[5],
            anisou.seg_id, anisou.element, anisou.charge);
        break;

    case ATOM:
        count = sprintf(buf,
"ATOM %6d %-4s%c%-4s%c%4d%c   %8.3f%8.3f%8.3f%6.2f%6.2f      %-4s%2s%-.2s",
            atom.serial, atom.name, atom.alt_loc, atom.res.name,
            atom.res.chain_id, atom.res.seq_num, atom.res.i_code,
            atom.xyz[0], atom.xyz[1], atom.xyz[2], atom.occupancy,
            atom.temp_factor, atom.seg_id, atom.element, atom.charge);
        break;

        case ATOMQR: 
                    count = sprintf(buf, 
    "ATOM %6d %-4s%c%-4s%c%4d%c   %8.3f%8.3f%8.3f %7.4f%7.4f", 
                            atomqr.serial, atomqr.name, atomqr.alt_loc, 
                            atomqr.res.name, atomqr.res.chain_id, 
                            atomqr.res.seq_num, atomqr.res.i_code, 
                            atomqr.xyz[0], atomqr.xyz[1], atomqr.xyz[2], 
                            atomqr.charge, atomqr.radius); 
                    break; 

    case AUTHOR:
        count = sprintf(buf, "AUTHOR  %2D%-.70s", author.continuation,
            author.author_list);
        break;

    case CAVEAT:
        count = sprintf(buf, "CAVEAT  %2D %-4s    %-.61s",
            caveat.continuation, caveat.id_code, caveat.comment);
        break;

    case CISPEP:
        count = sprintf(buf,
        "CISPEP %3d %-4s%c %4d%c   %-4s%c %4d%c       %3d       %6.2f",
            cispep.ser_num, cispep.pep[0].name,
            cispep.pep[0].chain_id, cispep.pep[0].seq_num,
            cispep.pep[0].i_code, cispep.pep[1].name,
            cispep.pep[1].chain_id, cispep.pep[1].seq_num,
            cispep.pep[1].i_code, cispep.mod_num, cispep.measure);
        break;

    case COMPND:
        count = sprintf(buf, "COMPND %3D%-.60s", compnd.continuation,
            compnd.compound);
        break;

    case CONECT:
        count = sprintf(buf, "CONECT%5d%5D%5D%5D%5D%5D%5D%5D%5D%5D%5D",
            conect.serial[0],
            conect.serial[1],
            conect.serial[2],
            conect.serial[3],
            conect.serial[4],
            conect.serial[5],
            conect.serial[6],
            conect.serial[7],
            conect.serial[8],
            conect.serial[9],
            conect.serial[10]);
        break;

    case CRYST1:
        count = sprintf(buf,
            "CRYST1%9.3f%9.3f%9.3f%7.2f%7.2f%7.2f %-11s%4d",
            cryst1.a, cryst1.b, cryst1.c,
            cryst1.alpha, cryst1.beta, cryst1.gamma,
            cryst1.s_group, cryst1.z);
        break;

    case DBREF:
        if (dbref.seq_begin2 == 0 && dbref.seq_end2 == 0)
            fmt = "DBREF  %-4s %c %4d%c %4d%c %-6s %-8s %-12s %5D%c %5D%c";
        else
            fmt = "DBREF  %-4s %c %4d%c %4d%c %-6s %-8s %-12s %5d%c %5d%c";
        count = sprintf(buf, fmt,
            dbref.id_code, dbref.chain_id, dbref.seq_begin,
            dbref.insert_begin, dbref.seq_end, dbref.insert_end,
            dbref.database, dbref.db_accession, dbref.db_id_code,
            dbref.seq_begin2, dbref.ins_beg_pdb, dbref.seq_end2,
            dbref.ins_end_pdb);
        break;

    case DBREF1:
        count = sprintf(buf,
            "DBREF1 %-4s %c %4D%c %4D%c %-6s               %-.20s",
            dbref1.id_code, dbref1.chain_id, dbref1.seq_begin,
            dbref1.insert_begin, dbref1.seq_end, dbref1.insert_end,
            dbref1.database, dbref1.db_id_code);
        break;

    case DBREF2:
        count = sprintf(buf,
        "DBREF2 %-4s %c     %-22s    %10D   %10D",
            dbref2.id_code, dbref2.chain_id, dbref2.db_accession,
            dbref2.seq_begin, dbref2.seq_end);
        break;

    case END:
        count = sprintf(buf, "END");
        break;

    case ENDMDL:
        count = sprintf(buf, "ENDMDL");
        break;

    case EXPDTA:
        count = sprintf(buf, "EXPDTA  %2D%-.60s", expdta.continuation,
            expdta.technique);
        break;

    case FORMUL:
        count = sprintf(buf, "FORMUL  %2d  %-4s%2D%c%-.51s",
            formul.comp_num, formul.het_id, formul.continuation,
            formul.exclude, formul.formula);
        break;

    case FTNOTE:
        // this record was dropped from PDB version 2.0
        // but still appears in RCSB PDB files
        count = sprintf(buf, "FTNOTE %3s %-.69s", ftnote.num,
            ftnote.text);
        break;

    case HEADER:
        count = sprintf(buf, "HEADER    %-40s%-9s   %-.4s",
            header.classification, header.dep_date, header.id_code);
        break;

    case HELIX:
        count = sprintf(buf,
            "HELIX  %3d %-3s %-4s%c %4d%c %-4s%c %4d%c%2d%-30s %5D",
            helix.ser_num, helix.helix_id, helix.init.name,
            helix.init.chain_id, helix.init.seq_num,
            helix.init.i_code, helix.end.name, helix.end.chain_id,
            helix.end.seq_num, helix.end.i_code, helix.helix_class,
            helix.comment, helix.length);
        break;

    case HET:
        count = sprintf(buf, "HET    %-4s %c%4d%c  %5d     %-.50s",
            het.res.name, het.res.chain_id, het.res.seq_num,
            het.res.i_code, het.num_het_atoms, het.text);
        break;

    case HETATM:
        count = sprintf(buf,
"HETATM%5d %-4s%c%-4s%c%4d%c   %8.3f%8.3f%8.3f%6.2f%6.2f      %-4s%2s%-.2s",
            hetatm.serial, hetatm.name, hetatm.alt_loc,
            hetatm.res.name, hetatm.res.chain_id, hetatm.res.seq_num,
            hetatm.res.i_code, hetatm.xyz[0], hetatm.xyz[1],
            hetatm.xyz[2], hetatm.occupancy, hetatm.temp_factor,
            hetatm.seg_id, hetatm.element, hetatm.charge);
        break;

    case HETNAM:
        count = sprintf(buf, "HETNAM  %2D %-4s%-.65s",
            hetnam.continuation, hetnam.het_id, hetnam.name);
        break;

    case HETSYN:
        count = sprintf(buf, "HETSYN  %2D %-4s%-.65s",
            hetsyn.continuation, hetsyn.het_id, hetsyn.synonyms);
        break;

    case HYDBND:
        count = sprintf(buf,
    "HYDBND      %-4s%c%-4s%c%5d%c %-4s%c %c%5D%c %-4s%c%-4s%c%5d%c%-6D %-6D",
            hydbnd.name[0], hydbnd.alt_loc[0], hydbnd.res[0].name,
            hydbnd.res[0].chain_id, hydbnd.res[0].seq_num,
            hydbnd.res[0].i_code,
            hydbnd.name[1], hydbnd.alt_loc[1],
            hydbnd.res[1].chain_id, hydbnd.res[1].seq_num,
            hydbnd.res[1].i_code,
            hydbnd.name[2], hydbnd.alt_loc[2], hydbnd.res[2].name,
            hydbnd.res[2].chain_id, hydbnd.res[2].seq_num,
            hydbnd.res[2].i_code,
            hydbnd.sym[0], hydbnd.sym[1]);
        break;

    case JRNL:
        count = sprintf(buf, "JRNL        %-.58s", jrnl.text);
        break;

    case KEYWDS:
        count = sprintf(buf, "KEYWDS  %2D%-.70s", keywds.continuation,
            keywds.keywds);
        break;

    case LINK:
        count = sprintf(buf,
"LINK        %-4s%c%-4s%c%4d%c               %-4s%c%-4s%c%4d%c  %6D %6D %5.2F",
            link.name[0], link.alt_loc[0], link.res[0].name,
            link.res[0].chain_id, link.res[0].seq_num,
            link.res[0].i_code,
            link.name[1], link.alt_loc[1], link.res[1].name,
            link.res[1].chain_id, link.res[1].seq_num,
            link.res[1].i_code,
            link.sym[0], link.sym[1], link.length);
        break;

    case MASTER:
        count = sprintf(buf,
            "MASTER    %5d%5d%5d%5d%5d%5d%5d%5d%5d%5d%5d%5d",
            master.num_remark, master.num_ftnote,
            master.num_het, master.num_helix, master.num_sheet,
            master.num_turn, master.num_site, master.num_xform,
            master.num_coord, master.num_ter, master.num_conect,
            master.num_seq);
        break;

    case MDLTYP:
        count = sprintf(buf, "MDLTYP  %2D%-.70s", mdltyp.continuation,
            mdltyp.comment);
        break;

    case MODEL:
        count = sprintf(buf, "MODEL %8d", model.serial);
        break;

    case MODRES:
        count = sprintf(buf, "MODRES %-4s %-4s%c %4d%c %-4s %-.51s",
            modres.id_code, modres.res.name, modres.res.chain_id,
            modres.res.seq_num, modres.res.i_code, modres.std_res,
            modres.comment);
        break;

    case MTRIX:
        count = sprintf(buf,
            "MTRIX%d %3d%10.6f%10.6f%10.6f     %10.5f    %D",
            mtrix.row_num, mtrix.serial, mtrix.m[0], mtrix.m[1],
            mtrix.m[2], mtrix.v, mtrix.i_given);
        break;

    case NUMMDL:
        count = sprintf(buf, "NUMMDL    %-4d", nummdl.model_number);
        break;

    case OBSLTE:
        count = sprintf(buf,
    "OBSLTE  %2D %-9s %-4s      %-4s %-4s %-4s %-4s %-4s %-4s %-4s %-4.s",
            obslte.continuation, obslte.rep_date, obslte.id_code,
            obslte.r_id_code[0], obslte.r_id_code[1],
            obslte.r_id_code[2], obslte.r_id_code[3],
            obslte.r_id_code[4], obslte.r_id_code[5],
            obslte.r_id_code[6], obslte.r_id_code[7]);
        break;

    case ORIGX:
        count = sprintf(buf, "ORIGX%d    %10.6f%10.6f%10.6f     %10.5f",
            origx.row_num, origx.o[0], origx.o[1], origx.o[2],
            origx.t);
        break;

    case REMARK:
        count = sprintf(buf, "REMARK %3d %-.69s", remark.remark_num,
            remark.empty);
        break;

    case REVDAT:
        count = sprintf(buf,
        "REVDAT %3d%2D %-9s %-5s   %1d       %-6s %-6s %-6s %-.6s",
            revdat.mod_num, revdat.continuation, revdat.mod_date,
            revdat.mod_id, revdat.mod_type, revdat.record[0],
            revdat.record[1], revdat.record[2], revdat.record[3]);
        break;

    case SCALE:
        count = sprintf(buf, "SCALE%d    %10.6f%10.6f%10.6f     %10.5f",
            scale.row_num, scale.s[0], scale.s[1], scale.s[2],
            scale.u);
        break;

    case SEQADV:
        count = sprintf(buf,
            "SEQADV %-4s %-4s%c %4d%c %-4s %-9s %-4s%5D %-.31s",
            seqadv.id_code, seqadv.res.name, seqadv.res.chain_id,
            seqadv.res.seq_num, seqadv.res.i_code, seqadv.database,
            seqadv.db_id_code, seqadv.db_res, seqadv.db_seq,
            seqadv.conflict);
        break;

    case SEQRES:
        count = sprintf(buf,
"SEQRES %3d %c %4d  %-4s%-4s%-4s%-4s%-4s%-4s%-4s%-4s%-4s%-4s%-4s%-4s%-.4s",
            seqres.ser_num, seqres.chain_id,
            seqres.num_res, seqres.res_name[0], seqres.res_name[1],
            seqres.res_name[2], seqres.res_name[3],
            seqres.res_name[4], seqres.res_name[5],
            seqres.res_name[6], seqres.res_name[7],
            seqres.res_name[8], seqres.res_name[9],
            seqres.res_name[10], seqres.res_name[11],
            seqres.res_name[12]);
        break;

    case SHEET:
        count = sprintf(buf,
"SHEET  %3d %-3s%2d %-4s%c%4d%c %-4s%c%4d%c%2d %-4s%-4s%c%4D%c %-4s%-4s%c%4D%c",
            sheet.strand, sheet.sheet_id, sheet.num_strands,
            sheet.init.name, sheet.init.chain_id, sheet.init.seq_num,
            sheet.init.i_code, sheet.end.name, sheet.end.chain_id,
            sheet.end.seq_num, sheet.end.i_code, sheet.sense,
            sheet.cur_atom, sheet.cur.name, sheet.cur.chain_id,
            sheet.cur.seq_num, sheet.cur.i_code, sheet.prev_atom,
            sheet.prev.name, sheet.prev.chain_id, sheet.prev.seq_num,
            sheet.prev.i_code);
        break;

    case SIGATM:
        count = sprintf(buf,
"SIGATM%5d %-4s%c%-4s%c%4d%c   %8.3f%8.3f%8.3f%6.2f%6.2f      %-4s%2s%-.2s",
            sigatm.serial, sigatm.name, sigatm.alt_loc,
            sigatm.res.name, sigatm.res.chain_id, sigatm.res.seq_num,
            sigatm.res.i_code, sigatm.sig_xyz[0], sigatm.sig_xyz[1],
            sigatm.sig_xyz[2], sigatm.sig_occ, sigatm.sig_temp,
            sigatm.seg_id, sigatm.element, sigatm.charge);
        break;

    case SIGUIJ:
        count = sprintf(buf,
        "SIGUIJ%5d %-4s%c%-4s%c%4d%c %7d%7d%7d%7d%7d%7d  %-4s%2s%-.2s",
            siguij.serial, siguij.name, siguij.alt_loc,
            siguij.res.name, siguij.res.chain_id, siguij.res.seq_num,
            siguij.res.i_code, siguij.sig[0], siguij.sig[1],
            siguij.sig[2], siguij.sig[3], siguij.sig[4],
            siguij.sig[5], siguij.seg_id, siguij.element,
            siguij.charge);
        break;

    case SITE:
        count = sprintf(buf, "SITE   %3d %-3s %2d",
                site.seq_num, site.site_id, site.num_res);
        for (int i = 0; i < 4; ++i) {
            if (site.res[i].name[0]) {
                count += sprintf(&buf[count], " %-4s%c%4d%c",
                    site.res[i].name, site.res[i].chain_id,
                    site.res[i].seq_num, site.res[i].i_code);
            } else {
                count += sprintf(&buf[count], "           ");
            }
        }
        break;

    case SLTBRG:
        count = sprintf(buf,
"SLTBRG      %-4s%c%-4s%c%4d%c               %-4s%c%-4s%c%4D%c  %-6D %-6D",
            sltbrg.name[0], sltbrg.alt_loc[0], sltbrg.res[0].name,
            sltbrg.res[0].chain_id, sltbrg.res[0].seq_num,
            sltbrg.res[0].i_code,
            sltbrg.name[1], sltbrg.alt_loc[1], sltbrg.res[1].name,
            sltbrg.res[1].chain_id, sltbrg.res[1].seq_num,
            sltbrg.res[1].i_code,
            sltbrg.sym[0], sltbrg.sym[1]);
        break;

    case SOURCE:
        count = sprintf(buf, "SOURCE %3D%-.60s", source.continuation,
            source.src_name);
        break;

    case SPLIT:
        count = sprintf(buf,
"SPLIT   %2D %-4s %-4s %-4s %-4s %-4s %-4s %-4s %-4s %-4s %-4s %-4s %-4s %-4s %-.4s",
            split.continuation,
            split.id_code[0], split.id_code[1], split.id_code[2], 
            split.id_code[3], split.id_code[4], split.id_code[5], 
            split.id_code[6], split.id_code[6], split.id_code[7], 
            split.id_code[9], split.id_code[10], split.id_code[11], 
            split.id_code[12], split.id_code[13]);
        break;

    case SPRSDE:
        count = sprintf(buf,
    "SPRSDE  %2D %-9s %-4s      %-4s %-4s %-4s %-4s %-4s %-4s %-4s %-.4s",
            sprsde.continuation, sprsde.sprsde_date,
            sprsde.id_code, sprsde.s_id_code[0], sprsde.s_id_code[1],
            sprsde.s_id_code[2], sprsde.s_id_code[3],
            sprsde.s_id_code[4], sprsde.s_id_code[5],
            sprsde.s_id_code[6], sprsde.s_id_code[7]);
        break;

    case SSBOND:
        count = sprintf(buf,
    "SSBOND %3d %-4s%c %4d%c   %-4s%c %4d%c                       %6D %6D %5.2F",
            ssbond.ser_num, ssbond.res[0].name,
            ssbond.res[0].chain_id, ssbond.res[0].seq_num,
            ssbond.res[0].i_code, ssbond.res[1].name,
            ssbond.res[1].chain_id, ssbond.res[1].seq_num,
            ssbond.res[1].i_code, ssbond.sym[0], ssbond.sym[1],
            ssbond.length);
        break;

    case TER:
        count = sprintf(buf, "TER   %5D      %-4s%c%4D%c", ter.serial,
            ter.res.name, ter.res.chain_id, ter.res.seq_num,
            ter.res.i_code);
        break;

    case TITLE:
        count = sprintf(buf, "TITLE   %2D%-.60s", title.continuation,
            title.title);
        break;

    case TURN:
        count = sprintf(buf,
            "TURN   %3d %-3s %-4s%c%4d%c %-4s%c%4d%c    %-.30s",
            turn.seq, turn.turn_id, turn.init.name,
            turn.init.chain_id, turn.init.seq_num, turn.init.i_code,
            turn.end.name, turn.end.chain_id, turn.end.seq_num,
            turn.end.i_code, turn.comment);
        break;

    case TVECT:
        count = sprintf(buf, "TVECT  %3d%10.5f%10.5f%10.5f%-.30s",
            tvect.serial, tvect.t[0], tvect.t[1], tvect.t[2],
            tvect.comment);
        break;

    case USER:
        count = sprintf(buf, "USER%-2s%-.74s", user.subtype, user.text);
        break;

    default:
        count = sprintf(buf, "unknown pdb record #%d", r_type);
        break;
    }

    // find last non-blank in buf, and shorten it
    while (count > 1 && isspace(buf[count - 1]))
        count -= 1;
    buf[count] = '\0';
    return buf;
}

}  // namespace pdb
