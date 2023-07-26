// vi: set expandtab ts=4 sw=4

/*
 * === UCSF ChimeraX Copyright ===
 * Copyright 2016 Regents of the University of California.
 * All rights reserved.  This software provided pursuant to a
 * license agreement containing restrictions on its disclosure,
 * duplication and use.  For details see:
 * http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
 * This notice must be embedded in or attached to all copies,
 * including partial copies, of the software or any revisions
 * or derivations thereof.
 * === UCSF ChimeraX Copyright ===
 */

#include <algorithm>
#include <exception>
#include <list>
#include <map>
#include <string>

#include <logger/logger.h>

#define ATOMSTRUCT_EXPORT
#define PYINSTANCE_EXPORT
#include "Atom.h"
#include "AtomicStructure.h"
#include "CompSS.h"
#include "Coord.h"
#include "Residue.h"
#include "search.h"

class bad_coords_error: public std::exception {
public:
    const char* what() const noexcept {
        return "Structure has degenerate atomic coordinates; assigning all 'turn' secondary structure";
    }
};

namespace atomstruct {

enum {
        DSSP_3DONOR           = 0x0001,
        DSSP_3ACCEPTOR        = 0x0002,
        DSSP_3GAP             = 0x0004,
        DSSP_3HELIX           = 0x0008,
        DSSP_4DONOR           = 0x0010,
        DSSP_4ACCEPTOR        = 0x0020,
        DSSP_4GAP             = 0x0040,
        DSSP_4HELIX           = 0x0080,
        DSSP_5DONOR           = 0x0100,
        DSSP_5ACCEPTOR        = 0x0200,
        DSSP_5GAP             = 0x0400,
        DSSP_5HELIX           = 0x0800,
        DSSP_PBRIDGE          = 0x1000,
        DSSP_ABRIDGE          = 0x2000,

        DSSP_PARA             = 1,
        DSSP_ANTI             = 2
};

using atomstruct::Atom;
using atomstruct::Coord;
using atomstruct::Real;
using atomstruct::Residue;
using atomstruct::Structure;

struct KsdsspCoords {
    const Coord *c, *n, *ca, *o, *h;
};

struct KsdsspLadderInfo {
    int    type;
    int    start[2], end[2];
    bool    is_bulge;
        KsdsspLadderInfo(int t, int s1, int e1, int s2, int e2) {
            type = t;
            if (s1 < e1) {
                start[0] = s1;
                end[0] = e1;
            } else {
                start[0] = e1;
                end[0] = s1;
            }
            if (s2 < e2) {
                start[1] = s2;
                end[1] = e2;
            } else {
                start[1] = e2;
                end[1] = s2;
            }
            is_bulge = false;
        }
};
// need to use a list here to allow erase() during iteration
typedef std::list<KsdsspLadderInfo> KsdsspLadders;
typedef std::vector<std::vector<bool>> KsdsspBool2D;
typedef std::vector<std::pair<int,int>> KsdsspHelices;

struct KsdsspParams {
    std::vector<Coord *>  imide_Hs;
    KsdsspBool2D  hbonds;
    float  hbond_cutoff;
    int  min_helix_length, min_strand_length;
    std::map<Residue *, int>  rflags;
    KsdsspLadders  ladders;
    KsdsspHelices  helices;
    std::vector<KsdsspCoords *>  coords;
    std::vector<Residue *>  residues;
    bool  report;
	CompSSInfo*	ss_info;
	AtomSearchTree* search_tree;
	std::map<Atom*, std::vector<Residue*>::size_type> search_lookup;
};

//
// Find the imide hydrogen position if it is missing
//
Coord *
add_imide_hydrogen(KsdsspCoords *cur, KsdsspCoords *prev)
{
    if (cur->h != nullptr)
        return nullptr;        // Already there

    const Coord n_coord = *cur->n;
    const Coord ca_coord = *cur->ca;
    const Coord c_coord = *prev->c;
    const Coord o_coord = *prev->o;

    auto n2ca = ca_coord - n_coord;
    auto n2c = c_coord - n_coord;
    auto c2o = o_coord - c_coord;
    n2ca.normalize();
    n2c.normalize();
    c2o.normalize();
    auto cac_bisect = n2ca + n2c;
    cac_bisect.normalize();
    auto opp_n = cac_bisect + c2o;
    opp_n.normalize();

    const Real nh_length = 1.01;
    Coord *h_coord = new Coord;
    *h_coord = n_coord - opp_n * nh_length;
    return h_coord;
}

//
// Add the imide hydrogens to all residues
//
static void
add_imide_hydrogens(KsdsspParams& params)
{
    int max = params.residues.size();
    if (max == 0)
        return;
    Residue *prev_res = params.residues[0];
    KsdsspCoords *prev_crd = params.coords[0];
    for (int i = 1; i < max; ++i) {
        Residue *r = params.residues[i];
        KsdsspCoords *crd = params.coords[i];
        if (r->connects_to(prev_res)) {
            Coord *h_coord = add_imide_hydrogen(crd, prev_crd);
            if (h_coord != nullptr) {
                params.imide_Hs.push_back(h_coord);
                crd->h = h_coord;
            }
        }
        prev_res = r;
        prev_crd = crd;
    }
}

//
// Check if other residue is hydrogen bonded to this one
//
static bool
hbonded_to(KsdsspCoords *c1, KsdsspCoords *other_crds, float hbond_cutoff)
{
    const Real q1 = 0.42;
    const Real q2 = 0.20;
    const Real f = 332;

    if (other_crds->h == nullptr)
        return false;
    const Coord *h_coord = other_crds->h;
    const Coord *c_coord = c1->c;
    const Coord *n_coord = other_crds->n;
    const Coord *o_coord = c1->o;

    auto rCN = c_coord->sqdistance(*n_coord);
    if (rCN > 49.0)        // Optimize a little bit
        return false;
    rCN = sqrt(rCN);
    auto rON = o_coord->distance(*n_coord);
    auto rCH = c_coord->distance(*h_coord);
    auto rOH = o_coord->distance(*h_coord);

    Real E = q1 * q2 * (1 / rON + 1 / rCH - 1 / rOH - 1 / rCN) * f;
    return E < hbond_cutoff;
}

//
// Find hydrogen bonds
//
static void
find_hbonds(KsdsspParams& params)
{
	// it's okay for loop vars to be unsigned here, since we don't subtract from them
    auto num_res = params.residues.size();
    std::vector<bool> is_pro;
    // mark prolines
    for (auto r: params.residues) {
        Atom *n = r->find_atom("N");
        if (n == nullptr) {
            is_pro.push_back(false);
            continue;
        }
        auto& nnb = n->neighbors();
        Atom *cd = r->find_atom("CD");
        is_pro.push_back(cd && std::find(nnb.begin(), nnb.end(), cd) != nnb.end());
    }
    for (decltype(num_res) i = 0; i < num_res; ++i) {
        KsdsspCoords *crds1 = params.coords[i];
		for (auto near: params.search_tree->search(*(crds1->n), 10.0)) {
			auto near_index = params.search_lookup[near];
			if (near_index <= i+1)
				continue;
            KsdsspCoords *crds2 = params.coords[near_index];
            // proline backbone nitrogen cannot donate
            if (is_pro[near_index])
                params.hbonds[i][near_index] = false;
            else
                params.hbonds[i][near_index] = hbonded_to(crds1, crds2, params.hbond_cutoff);
            if (is_pro[i])
                params.hbonds[near_index][i] = false;
            else
                params.hbonds[near_index][i] = hbonded_to(crds2, crds1, params.hbond_cutoff);
        }
    }
}

//
// Find the n-turns (n = 3,4,5)
//
static void
find_turns(KsdsspParams& params, int n)
{
    int donor = n == 3 ? DSSP_3DONOR :
            (n == 4 ? DSSP_4DONOR : DSSP_5DONOR);
    int acceptor = n == 3 ? DSSP_3ACCEPTOR : 
            (n == 4 ? DSSP_4ACCEPTOR : DSSP_5ACCEPTOR);
    int gap = n == 3 ? DSSP_3GAP : 
            (n == 4 ? DSSP_4GAP : DSSP_5GAP);
    int max = params.residues.size() - n;
    for (int i = 0; i < max; ++i)
        if (params.hbonds[i][i+n]) {
            params.rflags[params.residues[i]] |= acceptor;
            for (int j = 1; j < n; ++j)
                params.rflags[params.residues[i+j]] |= gap;
            params.rflags[params.residues[i+n]] |= donor;
        }
}

//
// Mark helices based on n-turn information
//
static void
mark_helices(KsdsspParams& params, int n)
{
    int acceptor = n == 3 ? DSSP_3ACCEPTOR :
            (n == 4 ? DSSP_4ACCEPTOR : DSSP_5ACCEPTOR);
    int helix = n == 3 ? DSSP_3HELIX :
            (n == 4 ? DSSP_4HELIX : DSSP_5HELIX);
    int max = params.residues.size() - n;
    for (int i = 1; i < max; ++i)
        if (params.rflags[params.residues[i-1]] & acceptor
        && params.rflags[params.residues[i]] & acceptor)
            for (int j = 0; j < n; ++j)
                params.rflags[params.residues[i+j]] |= helix;
}

//
// Construct helices based on marker information
//
static void
find_helices(KsdsspParams& params)
{
    // Criteria:  run of mostly same type; single-residue run of another type allowed;
    // two consecutive '>' (acceptor only) indicate start of new helix

    int max = params.residues.size();
    int first = -1;
    int cur_helix_type;
    int acc_only_run = 0;
    bool in_initial_acc_only = false;
    for (int i = 0; i < max; ++i) {
        int flags = params.rflags[params.residues[i]];
        int helix_type = 0;
        int helix_flags = 0;
        bool acc_only = false;
        if (flags & DSSP_3HELIX) {
            helix_type = 3; // 3-10
            helix_flags = DSSP_3ACCEPTOR | DSSP_3DONOR | DSSP_3GAP;
            acc_only = !(flags & DSSP_3DONOR);
        } else if (flags & (DSSP_4HELIX | DSSP_5HELIX)) {
            helix_type = 4; // alpha
            helix_flags = DSSP_4ACCEPTOR | DSSP_4DONOR | DSSP_4GAP
                | DSSP_5ACCEPTOR | DSSP_5DONOR | DSSP_5GAP;
            acc_only = (flags & DSSP_4ACCEPTOR) && !(flags & DSSP_4DONOR);
        }
        if (helix_type && (flags & helix_flags)) {
            if (first < 0) {
                first = i;
                cur_helix_type = helix_type;
                in_initial_acc_only = true; // don't break helix if X>>
            } else if (helix_type != cur_helix_type) {
                if (i - first >= params.min_helix_length)
                    params.helices.push_back(std::make_pair(first, i-1));
                first = i;
                cur_helix_type = helix_type;
                acc_only_run = 0;
            } else {
                in_initial_acc_only = in_initial_acc_only && acc_only;
            }
            if (in_initial_acc_only) {
                in_initial_acc_only = acc_only || (i == first);
            } else if (acc_only) {
                if (acc_only_run > 0) {
                    if (i-1 - first >= params.min_helix_length)
                        params.helices.push_back(std::make_pair(first, i-2));
                    first = i-1;
                    cur_helix_type = helix_type;
                    acc_only_run = 0;
                    in_initial_acc_only = true;
                } else {
                    acc_only_run++;
                }
            } else {
                acc_only_run = 0;
            }
        } else if (first >= 0) {
            if (i - first >= params.min_helix_length)
                params.helices.push_back(std::make_pair(first, i-1));
            first = -1;
            acc_only_run = 0;
        }
    }
    if (first >= 0) {
        if (max - first >= params.min_helix_length)
            params.helices.push_back(std::make_pair(first, max-1));
    }
}

//
// Check whether two ladders should merge to form a beta bulge
// We take advantage of some properties of how the ladders were generated:
//    start/end(0) < start/end(1)
//
// Beta-bulge as defined by K&S:
//    "a bulge-linked ladder consists of two (perfect)
//    ladders or bridges of the same type connected by
//    at most one extra residue on one strand and at most
//    four residues on the other strand."
//
static KsdsspLadderInfo *
merge_bulge(KsdsspLadderInfo &lr1, KsdsspLadderInfo &lr2)
{
    auto l1 = &lr1;
    auto l2 = &lr2;
    if (l1->type != l2->type)
        return nullptr;
    // Make sure that l1 precedes l2
    if (l1->start[0] > l2->start[0]) {
        KsdsspLadderInfo *tmp = l1;
        l1 = l2;
        l2 = tmp;
    }

    int d0 = l2->start[0] - l1->end[0];
    if (d0 < 0 || d0 > 4)
        return nullptr;
    int d1;
    if (l1->type == DSSP_PARA)
        d1 = l2->start[1] - l1->end[1];
    else
        d1 = l1->start[1] - l2->end[1];
    if (d1 < 0 || d1 > 4)
        return nullptr;
    if (d0 > 1 && d1 > 1)
        return nullptr;

    int s0 = l1->start[0];
    int e0 = l2->end[0];
    int s1, e1;
    if (l1->type == DSSP_PARA) {
        s1 = l1->start[1];
        e1 = l2->end[1];
    }
    else {
        s1 = l2->start[1];
        e1 = l1->end[1];
    }
    auto l = new KsdsspLadderInfo(l1->type, s0, e0, s1, e1);
    l->is_bulge = true;
    return l;
}

//
// Find beta-bulges and merge the ladders
//
static bool
find_beta_bulge(KsdsspParams& params)
{
    for (auto i1 = params.ladders.begin(); i1 != params.ladders.end(); ++i1) {
        auto& l1 = *i1;
        if (l1.is_bulge)
            continue;
        auto i2 = i1;
        for (++i2; i2 != params.ladders.end(); ++i2) {
            auto& l2 = *i2;
            if (l2.is_bulge)
                continue;
            auto l = merge_bulge(l1, l2);
            if (l != nullptr) {
                params.ladders.erase(i1);
                params.ladders.erase(i2);
                params.ladders.push_back(*l);
                delete l;
                return true;
            }
        }
    }
    return false;
}

//
// Find bridges
//
static void
find_bridges(KsdsspParams& params)
{
	// these loop-related variable need to be int, so that that var-1 can be negative, not a large positive
    int max = params.residues.size();

    // First we construct a matrix and mark the bridges
    typedef std::vector<std::vector<char> > Bridge;
    Bridge bridge;
    bridge.resize(max);
    for (auto& br: bridge)
        br.resize(max);

    decltype(max) i;
    for (i = 0; i < max-1; ++i) {
		// we're looking for hbonds involving adjacent residues, so loosen search criteria
		for (auto near: params.search_tree->search(*(params.coords[i]->n), 20.0)) {
			int near_index = params.search_lookup[near];
			if (near_index <= i)
				continue;
            if ((i > 0 && params.hbonds[i-1][near_index] && params.hbonds[near_index][i+1])
            || (near_index < max-1 && params.hbonds[near_index-1][i] && params.hbonds[i][near_index+1])) {
                bridge[i][near_index] = 'P';
                params.rflags[params.residues[i]] |= DSSP_PBRIDGE;
                params.rflags[params.residues[near_index]] |= DSSP_PBRIDGE;
            }
            else if ((params.hbonds[i][near_index] && params.hbonds[near_index][i])
            || (i > 0 && near_index < max-1 && params.hbonds[i-1][near_index+1] && params.hbonds[near_index-1][i+1]))
            {
                bridge[i][near_index] = 'A';
                params.rflags[params.residues[i]] |= DSSP_ABRIDGE;
                params.rflags[params.residues[near_index]] |= DSSP_ABRIDGE;
            }
        }
    }

    // Now we loop through and find the ladders
    decltype(i) k;
    for (i = 0; i < max; ++i) {
        for (decltype(i) j = i + 1; j < max; ++j) {
            switch (bridge[i][j]) {
              case 'P':
                for (k = 0; i+k < max && j+k < max && bridge[i+k][j+k] == 'P'; ++k)
                    bridge[i+k][j+k] = 'p';
                k--;
                params.ladders.push_back(KsdsspLadderInfo(DSSP_PARA, i, i + k, j, j + k));
                break;
              case 'A':
                for (k = 0; i+k < max && j-k >= 0 && bridge[i+k][j-k] == 'A'; ++k) 
                    bridge[i+k][j-k] = 'a';
                k--;
                params.ladders.push_back(KsdsspLadderInfo(DSSP_ANTI, i, i + k, j - k, j));
                break;
            }
        }
    }

    // Now we merge ladders of beta-bulges
    while (find_beta_bulge(params))
        continue;

    // Finally we get rid of any ladder that is too short
    // (on either strand)
    KsdsspLadders::iterator li, next;
    for (li = params.ladders.begin(); li != params.ladders.end(); li = next) {
        next = li; next++;
        KsdsspLadderInfo &l = *li;
        if (l.end[0] - l.start[0] + 1 < params.min_strand_length
        || l.end[1] - l.start[1] + 1 < params.min_strand_length) {
            params.ladders.erase(li);
        }
    }
}

static void
make_summary(KsdsspParams& params)
{
    if (params.residues.size() < 2)
        return;
    auto logger = params.residues[0]->structure()->logger();
    
    logger::info(logger, "Helix Summary");
    for (auto start_end: params.helices) {
        Residue *start = params.residues[start_end.first];
        Residue *end = params.residues[start_end.second];
        logger::info(logger, start->str(), " -> ", end->str());
    }
    logger::info(logger, ""); // blank line

    logger::info(logger, "Ladder Summary");
    for (auto l: params.ladders) {
        Residue *s0 = params.residues[l.start[0]];
        Residue *s1 = params.residues[l.start[1]];
        Residue *e0 = params.residues[l.end[0]];
        Residue *e1 = params.residues[l.end[1]];
        logger::info(logger, s0->str(), " -> ", e0->str(), " ",
            (l.type == DSSP_PARA ? "parallel" : "antiparallel"),
            " ", s1->str(), " -> ", e1->str());
    }
    logger::info(logger, ""); // blank line

    // merge ladders to make sheets
    std::vector<std::set<int>> sheets;
    for (auto l: params.ladders) {
        std::set<int> strand_pair;
        int t0, t1;
        for (int i = 0; i < 2; ++i) {
            if (l.start[i] < l.end[i]) {
                t0 = l.start[i];
                t1 = l.end[i];
            } else {
                t0 = l.end[i];
                t1 = l.start[i];
            }
            for (int j = t0; j < t1+1; ++j) {
                strand_pair.insert(j);
            }
        }
        std::vector<std::vector<std::set<int>>::iterator> merge_with;
        for (auto i = sheets.begin(); i != sheets.end(); ++i) {
            auto& sheet = *i;
            for (auto si = strand_pair.begin(); si != strand_pair.end(); ++si) {
                if (sheet.find(*si) != sheet.end()) {
                    merge_with.push_back(i);
                    break;
                }
            }
        }
        while (merge_with.size()) {
            auto mi = merge_with.back();
            strand_pair.insert((*mi).begin(), (*mi).end());
            sheets.erase(mi);
            merge_with.pop_back();
        }
        sheets.push_back(strand_pair);
    }

    // make mapping from residues to sheet letters
    char sheet_let = 'A';
    std::map<Residue *, char> sheet_letters;
    for (auto& sheet: sheets) {
        for (auto ri: sheet)
            sheet_letters.insert(std::pair<Residue *, char>(params.residues[ri], sheet_let));
        if (sheet_let == 'Z')
            sheet_let = 'a';
        else if (sheet_let == 'z')
            sheet_let = 'A';
        else
            sheet_let++;
    }

    logger::info(logger, "Residue Summary");
    logger::html_info(logger, "<pre>");
    for (auto r: params.residues) {
        int rflags = params.rflags[r];
        char summary = ' ';
        if (rflags & (DSSP_3HELIX))
            summary = 'G';
        else if (rflags & (DSSP_4HELIX))
            summary = 'H';
        else if (rflags & (DSSP_5HELIX))
            summary = 'I';
        else if (rflags & (DSSP_PBRIDGE | DSSP_ABRIDGE))
            summary = 'E';

        char turn3 = ' ';
        if ((rflags & DSSP_3DONOR) && (rflags & DSSP_3ACCEPTOR))
            turn3 = 'X';
        else if (rflags & (DSSP_3ACCEPTOR))
            turn3 = '>';
        else if (rflags & (DSSP_3DONOR))
            turn3 = '<';
        else if (rflags & (DSSP_3GAP))
            turn3 = '3';

        char turn4 = ' ';
        if ((rflags & DSSP_4DONOR) && (rflags & DSSP_4ACCEPTOR))
            turn4 = 'X';
        else if (rflags & (DSSP_4ACCEPTOR))
            turn4 = '>';
        else if (rflags & (DSSP_4DONOR))
            turn4 = '<';
        else if (rflags & (DSSP_4GAP))
            turn4 = '4';

        char turn5 = ' ';
        if ((rflags & DSSP_5DONOR) && (rflags & DSSP_5ACCEPTOR))
            turn5 = 'X';
        else if (rflags & (DSSP_5ACCEPTOR))
            turn5 = '>';
        else if (rflags & (DSSP_5DONOR))
            turn5 = '<';
        else if (rflags & (DSSP_5GAP))
            turn5 = '5';

        char bridge = ' ';
        if ((rflags & DSSP_PBRIDGE) && (rflags & DSSP_ABRIDGE))
            bridge = '+';
        else if (rflags & (DSSP_PBRIDGE))
            bridge = 'p';
        else if (rflags & (DSSP_ABRIDGE))
            bridge = 'A';

        char sheet = ' ';
        if (sheet_letters.find(r) != sheet_letters.end())
            sheet = sheet_letters[r];

        auto rstr = r->str();
        while (rstr.size() < 7)
            rstr += " ";
        logger::html_info(logger, rstr, " ", summary,
            " ", turn3, " ", turn4, " ", turn5, " ", bridge, " ", sheet);
    }
    logger::html_info(logger, "</pre>");
}

static void
merge_ladder(int start, int end, std::map<int, int>& res_to_strand, std::map<int,
	std::set<int>>& strands, int* strand_num)
{
	int start_i, end_i;
	if (end > start) {
		start_i = start;
		end_i = end;
	} else {
		start_i = end;
		end_i = start;
	}
	std::set<int> component_strands;
	for (int i = start_i; i <= end_i; ++i) {
		auto s_i = res_to_strand.find(i);
		if (s_i != res_to_strand.end())
			component_strands.insert(s_i->second);
	}
	if (component_strands.empty()) {
		// create new strand
		std::set<int> strand;
		for (int i = start_i; i <= end_i; ++i) {
			strand.insert(i);
			res_to_strand[i] = *strand_num;
		}
		strands[*strand_num] = strand;
		*strand_num += 1;
	} else if (component_strands.size() == 1) {
		// just (possibly) increase the existing strand
		int strand_i = *component_strands.begin();
		for (int i = start_i; i <= end_i; ++i) {
			strands[strand_i].insert(i);
			res_to_strand[i] = strand_i;
		}
	} else {
		// create new strand and coalesce old strands and ladder into it
		std::set<int> strand;
		for (auto strand_i: component_strands) {
			for (auto i: strands[strand_i]) {
				strand.insert(i);
				res_to_strand[i] = *strand_num;
			}
			strands.erase(strand_i);
		}
		for (int i = start_i; i <= end_i; ++i) {
			strand.insert(i);
			res_to_strand[i] = *strand_num;
		}
		strands[*strand_num] = strand;
		*strand_num += 1;
	}
}

static void
fill_in_ss_info(KsdsspParams& params)
{
	// merge the various sides of ladders into strands
	std::map<int, int> res_to_strand;
	std::map<int, std::set<int>> strands;
	int strand_num = 0;
    for (auto l: params.ladders) {
		merge_ladder(l.start[0], l.end[0], res_to_strand, strands, &strand_num);
		merge_ladder(l.start[1], l.end[1], res_to_strand, strands, &strand_num);
    }
	std::map<int, int> strand_to_output_index;
	for (auto sn_srs: strands) {
		auto& res_indices = sn_srs.second;
		strand_to_output_index[sn_srs.first] = params.ss_info->strands.size();
		params.ss_info->strands.push_back(std::pair<Residue*, Residue*>(
			params.residues[*std::min_element(res_indices.begin(), res_indices.end())],
			params.residues[*std::max_element(res_indices.begin(), res_indices.end())]
		));
	}

	// form sheets from strands
	std::map<int, int> strand_to_sheet;
	std::map<int, std::set<int>> sheets;
	int sheet_num = 0;
	for (auto l: params.ladders) {
		auto strand1 = res_to_strand[l.start[0]];
		auto strand2 = res_to_strand[l.start[1]];
		auto sh1_i = strand_to_sheet.find(strand1);
		auto sh2_i = strand_to_sheet.find(strand2);
		if (sh1_i == strand_to_sheet.end() && sh2_i == strand_to_sheet.end()) {
			// start new sheet
			sheet_num += 1;
			sheets[sheet_num] = std::set<int>{strand1, strand2};
			strand_to_sheet[strand1] = sheet_num;
			strand_to_sheet[strand2] = sheet_num;
		} else if (sh1_i != strand_to_sheet.end() && sh2_i != strand_to_sheet.end()) {
			if (sh1_i->second != sh2_i->second) {
				// ladder joins different sheets; merge
				sheet_num += 1;
				std::set<int> sheet;
				for (auto sn: std::vector<int>{sh1_i->second, sh2_i->second}) {
					for (auto strand_i: sheets[sn]) {
						sheet.insert(strand_i);
						strand_to_sheet[strand_i] = sheet_num;
					}
					sheets.erase(sn);
				}
			}
			// else: ladder within same sheet, no need to do anything
		} else {
			// one strand in sheet and the other isn't in any sheet; add the other one
			int sheet;
			int other_strand;
			if (sh1_i == strand_to_sheet.end()) {
				sheet = sh2_i->second;
				other_strand = strand1;
			} else {
				sheet = sh1_i->second;
				other_strand = strand2;
			}
			strand_to_sheet[other_strand] = sheet;
			sheets[sheet].insert(other_strand);
		}
	}
	for (auto sheet_strands: sheets) {
		std::set<int> output_strands;
		for (auto sn: sheet_strands.second) {
			output_strands.insert(strand_to_output_index[sn]);
		}
		params.ss_info->sheets.push_back(output_strands);
	}

	// provide (anti-)parallel strand info
    for (auto l: params.ladders) {
		auto strand1 = res_to_strand[l.start[0]];
		auto strand2 = res_to_strand[l.start[1]];
		params.ss_info->strands_parallel[std::pair<int, int>(strand1, strand2)] = l.type == DSSP_PARA;
		params.ss_info->strands_parallel[std::pair<int, int>(strand2, strand1)] = l.type == DSSP_PARA;
    }

	// helix information
    for (auto start_end: params.helices) {
        Residue *start = params.residues[start_end.first];
        Residue *end = params.residues[start_end.second];
        int rflags = params.rflags[start];
        char summary = 'H';
        if (rflags & (DSSP_3HELIX))
            summary = 'G';
        else if (rflags & (DSSP_5HELIX))
            summary = 'I';
		params.ss_info->helix_info.push_back({{start, end}, summary});
    }
}

static void
compute_chain(KsdsspParams& params)
{
    int num_res = params.residues.size();
    params.hbonds.resize(num_res);
    for (auto& hbonds: params.hbonds)
        hbonds.resize(num_res);

    // Compute secondary structure
    try {
        add_imide_hydrogens(params);
    } catch (std::domain_error&) {
        throw bad_coords_error();
    }
    find_hbonds(params);

    find_turns(params, 3);
    mark_helices(params, 3);
    find_turns(params, 4);
    mark_helices(params, 4);
    find_turns(params, 5);
    mark_helices(params, 5);
    find_helices(params);

    find_bridges(params);
    // Don't need to find entire sheets per se, pairs of strands
    // (i.e. ladders) are good enough

    // actually markup the structure
    // do some fancy footwork to ensure that strands are numbered
    // in N->C order
    std::vector<std::pair<int, int> > res_ranges;
    for (auto l: params.ladders) {
        res_ranges.push_back(std::pair<int,int>(l.start[0], l.end[0]));
        res_ranges.push_back(std::pair<int,int>(l.start[1], l.end[1]));
    }
    int id = 0;
    int last = -1;
    std::sort(res_ranges.begin(), res_ranges.end());
    for (auto start_end: res_ranges) {
        if (start_end.first > last)
            ++id;
        for (int i = start_end.first; i <= start_end.second; ++i) {
            Residue *r = params.residues[i];
            r->set_is_strand(true);
            r->set_ss_id(id);
        }
        last = start_end.second;
    }

    id = 0;
    for (auto start_end: params.helices) {
        id++;
        for (int i = start_end.first; i <= start_end.second; ++i) {
            Residue *r = params.residues[i];
            r->set_is_helix(true);
            r->set_ss_id(id);
        }
    }

    if (params.report)
        make_summary(params);

	if (params.ss_info != nullptr)
		fill_in_ss_info(params);
}

void
AtomicStructure::compute_secondary_structure(float energy_cutoff,
    int min_helix_length, int min_strand_length, bool report, CompSSInfo* ss_info)
{
    // initialize
    KsdsspParams params;
    try {
        params.hbond_cutoff = energy_cutoff;
        params.min_helix_length = min_helix_length;
        params.min_strand_length = min_strand_length;
        params.report = report;
		params.ss_info = ss_info;
        // commented out lines that restricted
        // sheets to be intra-chain
        //Residue *prev_res = nullptr;
		std::vector<Atom*> ns;
        for (auto r: residues()) {
            r->set_is_helix(false);
            r->set_is_strand(false);
            r->set_ss_id(-1);
            //if (prev_res && !r->connects_to(prev_res)) {
            //    compute_chain(info);
            //}
            //prev_res = r;
            Atom *c = r->find_atom("C");
            if (!c)
                continue;
            Atom *n = r->find_atom("N");
            Atom *ca = r->find_atom("CA");
            Atom *o = r->find_atom("O");
            if (!n || !ca || !o)
                continue;
            

			ns.push_back(n);
			params.search_lookup[n] = params.residues.size();
            params.residues.push_back(r);
            KsdsspCoords *crds = new KsdsspCoords;
            params.coords.push_back(crds);

            crds->c = &c->coord();
            crds->n = &n->coord();
            crds->o = &o->coord();
            crds->ca = &ca->coord();

            Atom *h = r->find_atom("H");
            if (h)
                crds->h = &h->coord();
            else
                crds->h = nullptr;
        }
		AtomSearchTree ast(ns, false, 10.0);
		params.search_tree = &ast;
        compute_chain(params);
        set_ss_assigned(true);
        ss_ids_normalized = false;
        for (auto crd: params.coords)
            delete crd;
        for (auto ih: params.imide_Hs)
            delete ih;
    } catch (bad_coords_error& e) {
        set_ss_assigned(true); // leave as all-turn; don't try again
        for (auto crd: params.coords)
            delete crd;
        for (auto ih: params.imide_Hs)
            delete ih;
        logger::error(logger(), e.what());
    } catch (...) {
        set_ss_assigned(true); // leave as all-turn; don't try again
        for (auto crd: params.coords)
            delete crd;
        for (auto ih: params.imide_Hs)
            delete ih;
        throw;
    }
}

}  // namespace atomstruct
