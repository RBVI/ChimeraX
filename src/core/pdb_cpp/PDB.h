// vi: set expandtab ts=4 sw=4:
#ifndef pdb_PDB
#define pdb_PDB

#include <iostream>
#include "imex.h"

namespace pdb {

class PDB_IMEX PDB {
public:
    static const int BUF_LEN = 82;      // PDB record length (80 + "\n")
    //
    // These types are from PDB 2.0 Appendix 6.
    // The types that map to C types or types that map to character
    //  arrays of varying lengths are not duplicated.
    //
    typedef char    Atom[5];
    typedef char    Date[10];
    typedef char    IDcode[5];
    typedef double  Real;
    typedef char    ResidueName[5];     // local extension

    // our constructed type
    struct Residue {
        ResidueName  name;
        char  chain_id;
        int  seq_num;
        char  i_code;
    };

    // structures declarations for each record type

    typedef struct {
        char    junk[BUF_LEN];
    } Unknown_;
    typedef struct {
        int serial;
        Atom    name;
        char    alt_loc;
        Residue res;
        int u[6];
        char    seg_id[5];
        char    element[3];
        char    charge[3];
    } Anisou_;
    typedef struct {
        int serial;
        Atom    name;
        char    alt_loc;
        Residue res;
        Real    xyz[3];
        Real    occupancy, temp_factor;
        char    seg_id[5];
        char    element[3];
        char    charge[3];
    } Atom_;
    typedef struct {
        int serial;
        Atom    name;
        char    alt_loc;
        Residue res;
        Real    xyz[3];
        Real    charge, radius;
    } Atomqr_;
    typedef struct {
        int continuation;
        char    author_list[71];
    } Author_;
    typedef struct {
        int continuation;
        IDcode  id_code;
        char    comment[62];
    } Caveat_;
    typedef struct {
        int ser_num;
        Residue pep[2];
        int mod_num;
        Real    measure;
    } Cispep_;
    typedef struct {
        int continuation;
        char    compound[61];
    } Compnd_;
    typedef struct {
        int serial[11];
    } Conect_;
    typedef struct {
        Real    a, b, c;
        Real    alpha, beta, gamma;
        char    s_group[12];
        int z;
    } Cryst1_;
    typedef struct {
        IDcode  id_code;
        char    chain_id;
        int seq_begin;
        char    insert_begin;
        int seq_end;
        char    insert_end;
        char    database[7];
        char    db_accession[9];
        char    db_id_code[13];
        int seq_begin2;
        char    ins_beg_pdb;
        int seq_end2;
        char    ins_end_pdb;
    } Dbref_;
    typedef struct {
        IDcode  id_code;
        char    chain_id;
        int seq_begin;
        char    insert_begin;
        int seq_end;
        char    insert_end;
        char    database[7];
        char    db_id_code[21];
    } Dbref1_;
    typedef struct {
        IDcode  id_code;
        char    chain_id;
        char    db_accession[23];
        int seq_begin;
        int seq_end;
    } Dbref2_;
    // no structure for END
    // no structure for ENDMDL
    typedef struct {
        int continuation;
        char    technique[61];
    } Expdta_;
    typedef struct {
        int comp_num;
        ResidueName het_id;
        int continuation;
        char    exclude;    // '*' to exclude
        char    formula[52];
    } Formul_;
    typedef struct {        // removed in PDB Version 2.0
        char    num[4];
        char    text[70];
    } Ftnote_;
    typedef struct {
        char    classification[41];
        Date    dep_date;
        IDcode  id_code;
    } Header_;
    typedef struct {
        int ser_num;
        char    helix_id[4];
        Residue init;
        Residue end;
        int helix_class;
        char    comment[31];
        int length;
    } Helix_;
    typedef struct {
        Residue res;
        int num_het_atoms;
        char    text[51];
    } Het_;
    typedef Atom_   Hetatm_;
    typedef struct {
        int continuation;
        ResidueName het_id;
        char    name[66];
    } Hetnam_;
    typedef struct {
        int continuation;
        ResidueName het_id;
        char    synonyms[66];
    } Hetsyn_;
    typedef struct {
        // 0 = atom 1, 1 = atom 2, 2 = hydrogen atom
        Atom    name[3];
        char    alt_loc[3];
        Residue res[3];
        int sym[2];
    } Hydbnd_;
    typedef struct {
        int continuation;
        char    text[59];
    } Jrnl_;
    typedef struct {
        int continuation;
        char    keywds[71];
    } Keywds_;
    typedef struct {
        Atom    name[2];
        char    alt_loc[2];
        Residue res[2];
        int sym[2];
        Real    length;
    } Link_;
    typedef struct {
        int num_remark;
        int num_ftnote;
        int num_het;
        int num_helix;
        int num_sheet;
        int num_turn;
        int num_site;
        int num_xform;
        int num_coord;
        int num_ter;
        int num_conect;
        int num_seq;
    } Master_;
    typedef struct {
        int continuation;
        char    comment[71];
    } Mdltyp_;
    typedef struct {
        int serial;
    } Model_;
    typedef struct {
        IDcode  id_code;
        Residue res;
        ResidueName std_res;
        char    comment[52];
    } Modres_;
    // Mtrix_ is for MTRIX1, MTRIX2, and MTRIX3
    typedef struct {
        int row_num;
        int serial;
        Real    m[3], v;
        int i_given;
    } Mtrix_;
    typedef struct {
        int model_number;
    } Nummdl_;
    typedef struct {
        int continuation;
        Date    rep_date;
        IDcode  id_code;
        IDcode  r_id_code[8];
    } Obslte_;
    // Origx_ is for ORIGX1, ORIGX2, and ORIGX3
    typedef struct {
        int row_num;
        Real    o[3], t;
    } Origx_;
    typedef struct {
        int remark_num;
        char    empty[70];
    } Remark_;
    typedef struct {
        int mod_num;
        int continuation;
        Date    mod_date;
        char    mod_id[6];
        int mod_type;
        char    record[4][7];
    } Revdat_;
    // Scale_ is for SCALE1, SCALE2, and SCALE3
    typedef struct {
        int row_num;
        Real    s[3], u;
    } Scale_;
    typedef struct {
        IDcode  id_code;
        Residue res;
        char    database[5];
        char    db_id_code[10];
        ResidueName db_res;
        int db_seq;
        char    conflict[32];
    } Seqadv_;
    typedef struct {
        int ser_num;
        char    chain_id;
        int num_res;
        ResidueName res_name[13];
    } Seqres_;
    typedef struct {
        int strand;
        char    sheet_id[4];
        int num_strands;
        Residue init;
        Residue end;
        int sense;
        Atom    cur_atom;
        Residue cur;
        Atom    prev_atom;
        Residue prev;
    } Sheet_;
    typedef struct {
        int serial;
        Atom    name;
        char    alt_loc;
        Residue res;
        Real    sig_xyz[3];
        Real    sig_occ, sig_temp;
        char    seg_id[5];
        char    element[3];
        char    charge[3];
    } Sigatm_;
    typedef struct {
        int serial;
        Atom    name;
        char    alt_loc;
        Residue res;
        int sig[6];
        char    seg_id[5];
        char    element[3];
        char    charge[3];
    } Siguij_;
    typedef struct {
        int seq_num;
        char    site_id[4];
        int num_res;
        Residue res[4];
    } Site_;
    typedef struct {
        Atom    name[2];
        char    alt_loc[2];
        Residue res[2];
        int sym[2];
    } Sltbrg_;
    typedef struct {
        int continuation;
        char    src_name[61];
    } Source_;
    typedef struct {
        int continuation;
        IDcode  id_code[14];
    } Split_;
    typedef struct {
        int continuation;
        Date    sprsde_date;
        IDcode  id_code;
        IDcode  s_id_code[8];
    } Sprsde_;
    typedef struct {
        int ser_num;
        Residue res[2];
        int sym[2];
        Real    length;
    } Ssbond_;
    typedef struct {
        int serial;
        Residue res;
    } Ter_;
    typedef struct {
        int continuation;
        char    title[61];
    } Title_;
    typedef struct {
        int seq;
        char    turn_id[4];
        Residue init;
        Residue end;
        char    comment[31];
    } Turn_;
    typedef struct {
        int serial;
        Real    t[3];
        char    comment[31];
    } Tvect_;
    typedef struct {
        char    subtype[3];
        char    text[75];
    } User_;

    enum RecordType { UNKNOWN,
        ANISOU, ATOM, ATOM1, ATOM2, ATOM3, ATOM4, ATOM5, ATOM6, ATOM7,
        ATOM8, ATOM9, ATOMQR, AUTHOR,
        CAVEAT, CISPEP, COMPND, CONECT, CRYST1,
        DBREF, DBREF1, DBREF2, END, ENDMDL, EXPDTA, FORMUL, FTNOTE,
        HEADER, HELIX, HET, HETATM, HETNAM, HETSYN, HYDBND,
        JRNL, KEYWDS, LINK,
        MASTER, MDLTYP, MODEL, MODRES, MTRIX, NUMMDL,
        OBSLTE, ORIGX, REMARK, REVDAT,
        SCALE, SEQADV, SEQRES, SHEET, SIGATM, SIGUIJ, SITE, SLTBRG,
        SOURCE, SPLIT, SPRSDE, SSBOND,
        TER, TITLE, TURN, TVECT, USER
    };
    static const int    NUM_TYPES = EXPDTA + 1;

private:
    RecordType  r_type;
public:
    union {
        Unknown_    unknown;
        Anisou_ anisou;
        Atom_   atom;
        Atomqr_ atomqr; // PDBQR ATOM record
        Author_ author;
        Caveat_ caveat;
        Cispep_ cispep;
        Compnd_ compnd;
        Conect_ conect;
        Cryst1_ cryst1;
        // no End_ structure
        // no Endmdl_ structure
        Dbref_  dbref;
        Dbref1_ dbref1;
        Dbref2_ dbref2;
        Expdta_ expdta;
        Formul_ formul;
        Ftnote_ ftnote;
        Header_ header;
        Helix_  helix;
        Het_    het;
        Hetatm_ hetatm;
        Hetnam_ hetnam;
        Hetsyn_ hetsyn;
        Hydbnd_ hydbnd;
        Jrnl_   jrnl;
        Keywds_ keywds;
        Link_   link;
        Master_ master;
        Mdltyp_ mdltyp;
        Model_  model;
        Modres_ modres;
        Mtrix_  mtrix;
        Nummdl_ nummdl;
        Obslte_ obslte;
        Origx_  origx;
        Remark_ remark;
        Revdat_ revdat;
        Scale_  scale;
        Seqres_ seqres;
        Seqadv_ seqadv;
        Sheet_  sheet;
        Sigatm_ sigatm;
        Siguij_ siguij;
        Site_   site;
        Sltbrg_ sltbrg;
        Source_ source;
        Split_  split;
        Sprsde_ sprsde;
        Ssbond_ ssbond;
        Ter_    ter;
        Title_  title;
        Turn_   turn;
        Tvect_  tvect;
        User_   user;
    };
private:
    static int  input_version;
    static int  atom_serial_number;
    static int  sigatm_serial_number;
    static int  byte_cmp(const PDB &l, const PDB &r);
public:

            PDB() { set_type(UNKNOWN); }
            PDB(RecordType t) { set_type(t); }
            PDB(const char *buf);
    RecordType  type() const { return r_type; }
    void        set_type(RecordType t);
    void        parse_line(const char *buf);
    const char  *c_str() const;
    static int  pdb_input_version() { return input_version; }
    static RecordType get_type(const char *buf);
    static int  sscanf(const char *, const char *, ...);
    static int  sprintf(char *, const char *, ...);
    static void reset_state();

    inline bool operator==(const PDB &r) const {
                if (r_type != r.r_type)
                    return 0;
                return byte_cmp(*this, r) == 0;
            }
    inline bool operator!=(const PDB &r) const {
                if (r_type != r.r_type)
                    return 1;
                return byte_cmp(*this, r) != 0;
            }

    friend std::istream &operator>>(std::istream &s, PDB &p);
};

inline std::ostream &
operator<<(std::ostream &s, const PDB &p)
{
    s << p.c_str();
    return s;
}

}  // namespace pdb

#endif  //  pdb_PDB
