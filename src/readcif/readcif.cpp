/*
 * Copyright (c) 2014 The Regents of the University of California.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *   1. Redistributions of source code must retain the above copyright
 *      notice, this list of conditions, and the following disclaimer.
 *   2. Redistributions in binary form must reproduce the above
 *      copyright notice, this list of conditions, and the following
 *      disclaimer in the documentation and/or other materials provided
 *      with the distribution.
 *   3. Redistributions must acknowledge that this software was
 *      originally developed by the UCSF Resource for Biocomputing,
 *      Visualization, and Informatics with support from the National
 *      Institute of General Medical Sciences, grant P41-GM103311.
 *
 *   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER "AS IS" AND ANY
 *   EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 *   PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OF THE UNIVERSITY
 *   OF CALIFORNIA BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 *   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 *   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 *   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 *   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 *   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#undef CR_IS_EOL	/* undef for ~2% speedup */
#define CASE_INSENSITIVE	/* undef for ~6% speedup */
// variations on stylized parsing
#define FIXED_LENGTH_ROWS
#define COMMENT_TERMINATED

#include "readcif.h"
#include <limits.h>
#ifdef CASE_INSENSITIVE
# include <ctype.h>
#endif
#include <string.h>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <sstream>

#if UCHAR_MAX > 255
# error "character size is too big for internal tables"
#endif

using std::string;
using std::vector;
using readcif::StringVector;

// mmCIF files are CIF 1.1 compliant except for how strings are encoded.
// For example, in mmCIF, BETA-MERCAPTOETHANOL, would be \B-mercaptoethanol,
// in CIF 1.1.
//
// TODO: check if audit_conform.dict_name is "mmcif_pdbx.dic" to see if
// we're parsing a mmcif file or not.

namespace {

#ifdef CR_IS_EOL
const int EndOfLine[UCHAR_MAX + 1] = {
	// ASCII LF (10) and CR (13)
	// are the end-of-line characters recognized in CIF files
	// Also treat ASCII NUL (0) as an end of line terminator
	true, false, false, false, false, false, false, false,	// 0-7
	false, false, true, false, false, true, false, false,	// 8-15
	// the rest defaults to false
};
#endif

inline int
is_eol(char c)
{
#ifdef CR_IS_EOL
	return EndOfLine[(unsigned char) c];
#else
	return c == 0 || c == '\n';
#endif
}

#ifdef CR_IS_EOL
const int NotEndOfLine[UCHAR_MAX + 1] = {
	// treat ASCII NUL (0) as an end of line terminator
	false, true, true, true, true, true, true, true,	// 0-7
	true, true, false, true, true, false, true, true,	// 8-15
	true, true, true, true, true, true, true, true,		// 16-23
	true, true, true, true, true, true, true, true,		// 24-31
	true, true, true, true, true, true, true, true,		// 32-39
	true, true, true, true, true, true, true, true,		// 40
	true, true, true, true, true, true, true, true,		// 48
	true, true, true, true, true, true, true, true,		// 56
	true, true, true, true, true, true, true, true,		// 64
	true, true, true, true, true, true, true, true,		// 72
	true, true, true, true, true, true, true, true,		// 80
	true, true, true, true, true, true, true, true,		// 88
	true, true, true, true, true, true, true, true,		// 96
	true, true, true, true, true, true, true, true,		// 104
	true, true, true, true, true, true, true, true,		// 112
	true, true, true, true, true, true, true, true,		// 120
	true, true, true, true, true, true, true, true,		// 128
	true, true, true, true, true, true, true, true,		// 136
	true, true, true, true, true, true, true, true,		// 144
	true, true, true, true, true, true, true, true,		// 152
	true, true, true, true, true, true, true, true,		// 160
	true, true, true, true, true, true, true, true,		// 168
	true, true, true, true, true, true, true, true,		// 176
	true, true, true, true, true, true, true, true,		// 184
	true, true, true, true, true, true, true, true,		// 192
	true, true, true, true, true, true, true, true,		// 200
	true, true, true, true, true, true, true, true,		// 208
	true, true, true, true, true, true, true, true,		// 216
	true, true, true, true, true, true, true, true,		// 224
	true, true, true, true, true, true, true, true,		// 232
	true, true, true, true, true, true, true, true,		// 240
	true, true, true, true, true, true, true, true,		// 248-255
};
#endif

inline int
is_not_eol(char c)
{
#ifdef CR_IS_EOL
	return NotEndOfLine[(unsigned char) c];
#else
	return c && c != '\n';
#endif
}

#define STRNEQ_P1(name, buf, len) (strncmp((name) + 1, (buf) + 1, (len) - 1) == 0)
#ifndef CASE_INSENSITIVE
#define ICASEEQN_P1(name, buf, len) STRNEQ_P1(name, buf, len)
#else
// icaseeqn: 
// 	compare name in a case independent way to buf
bool
icaseeqn(const char *name, const char *buf, size_t len)
{
	for (size_t i = 0; i < len; ++i) {
		if (name[i] == '\0' || buf[i] == '\0')
			return name[i] == buf[i];
		if (tolower(name[i]) != tolower(buf[i]))
			return false;
	}
	return true;
}

// frequently, it is known that the first character already matches
#define ICASEEQN_P1(name, buf, len) icaseeqn((name) + 1, (buf) + 1, (len) - 1)
#endif

string
unescape_mmcif(const string& s)
{
	// TODO: Undo PDB conventions
	return s;
}

} // private namespace

namespace readcif {

// character tables use int insteal of bool because it is faster
// (presumably because all accesses are aligned)

const int Whitespace[UCHAR_MAX + 1] = {
	// ASCII HT (9), LF (10), CR (13), and SPACE (32)
	// are the only whitespace characters recognized in CIF files
	false, false, false, false, false, false, false, false,	// 0-7
	false, true, true, false, false, true, false, false,	// 8-15
	false, false, false, false, false, false, false, false,	// 16-23
	false, false, false, false, false, false, false, false,	// 24-31
	true, false, false, false, false, false, false, false,	// 32-39
	// the rest defaults to false
};

const int NotWhitespace[UCHAR_MAX + 1] = {
	// ASCII HT (9), LF (10), CR (13), and SPACE (32)
	// are the only whitespace characters recognized in CIF files
	// Treat ASCII NUL (0) as an end of line terminator to
	// avoid separate testing for NUL.
	false, true, true, true, true, true, true, true,	// 0-7
	true, false, false, true, true, false, true, true,	// 8-15
	true, true, true, true, true, true, true, true,		// 16-23
	true, true, true, true, true, true, true, true,		// 24-31
	false, true, true, true, true, true, true, true,	// 32-39
	true, true, true, true, true, true, true, true,		// 40
	true, true, true, true, true, true, true, true,		// 48
	true, true, true, true, true, true, true, true,		// 56
	true, true, true, true, true, true, true, true,		// 64
	true, true, true, true, true, true, true, true,		// 72
	true, true, true, true, true, true, true, true,		// 80
	true, true, true, true, true, true, true, true,		// 88
	true, true, true, true, true, true, true, true,		// 96
	true, true, true, true, true, true, true, true,		// 104
	true, true, true, true, true, true, true, true,		// 112
	true, true, true, true, true, true, true, true,		// 120
	true, true, true, true, true, true, true, true,		// 128
	true, true, true, true, true, true, true, true,		// 136
	true, true, true, true, true, true, true, true,		// 144
	true, true, true, true, true, true, true, true,		// 152
	true, true, true, true, true, true, true, true,		// 160
	true, true, true, true, true, true, true, true,		// 168
	true, true, true, true, true, true, true, true,		// 176
	true, true, true, true, true, true, true, true,		// 184
	true, true, true, true, true, true, true, true,		// 192
	true, true, true, true, true, true, true, true,		// 200
	true, true, true, true, true, true, true, true,		// 208
	true, true, true, true, true, true, true, true,		// 216
	true, true, true, true, true, true, true, true,		// 224
	true, true, true, true, true, true, true, true,		// 232
	true, true, true, true, true, true, true, true,		// 240
	true, true, true, true, true, true, true, true,		// 248-255
};

CIFFile::CIFFile()
{
	reset_parse();
}

void
CIFFile::register_category(const std::string& category, ParseCategory callback, 
					const StringVector& dependencies)
{
	for (auto dep: dependencies) {
		if (categories.find(dep) != categories.end())
			continue;
		std::ostringstream os;
		os << "Missing dependency " << dep << " for category "
							<< category;
		throw std::logic_error(os.str());
	}
	if (callback) {
		categoryOrder.push_back(category);
		categories.emplace(category,
				   CategoryInfo(callback, dependencies));
	} else {
		// TODO: find category in categoryOrder
		// make sure none of the later categories depend on it
		throw std::runtime_error("not implemented");
		categories.erase(category);
	}
}

void
CIFFile::parse(const char *whole_file)
{
	this->whole_file = whole_file;
	try {
		if (parsing)
			throw error("Already parsing");
		reset_parse();
		parsing = true;
		// Check for CIF version
		if (line[0] == '#' && strncmp(line, "#\\#CIF_", 7) == 0) {
			// update lexical state
			pos = line + 3;
			const char* e = line + 7;
			for (; is_not_whitespace(*e); ++e)
				continue;
			version_ = string(pos, e - pos);
			pos = e;
			for (; is_not_eol(*pos); ++pos)
				continue;
		}
		internal_parse();
		parsing = false;
	} catch (std::exception &e) {
		parsing = false;
		throw;
	}
}

std::runtime_error
CIFFile::error(const std::string& text)
{
	std::ostringstream os;
	os << text << " on line " << lineno;
	return std::move(std::runtime_error(os.str()));
}

inline std::string
CIFFile::current_value()
{
	return string(current_value_start, current_value_end - current_value_start);
}

void
CIFFile::internal_parse(bool one_table)
{
	next_token();
	for (;;) {
		switch (current_token) {
		case T_DATA:
			if (stash.size() > 0)
				process_stash();
			seen.clear();
			set_PDBx_stylized(false);
			current_data_block = current_value();
			data_block(current_data_block);
			if (stylized_)
				stylized_next_keyword(true);
			else
				next_token();
			continue;
		case T_LOOP: {
			const char* loop_pos = pos - 5;
			current_category.clear();
			current_tags.clear();
			values.clear();
			next_token();
			if (current_token != T_TAG)
				throw error("expected data name after loop_");
			Categories::iterator cii;
			string cv = current_value();
			size_t sep = cv.find('.');
			DDL_v2 = (sep != std::string::npos);
			if (DDL_v2) {
				current_category = cv.substr(0, sep);
				cii = categories.find(current_category);
			} else {
				cii = categories.end();
				current_category = cv;
				for (;;) {
					sep = current_category.rfind('_');
					if (sep == std::string::npos)
						break;
					current_category.resize(sep);
					cii = categories.find(current_category);
					if (cii != categories.end())
						break;
				}
			}
			bool keep = cii != categories.end();
			if (keep && !one_table) {
				for (auto d: cii->second.dependencies) {
					if (seen.find(d) != seen.end())
						continue;
					keep = false;
					stash.emplace(current_category,
					      StashInfo(loop_pos, lineno));
					break;
				}
			}
			if (keep) {
				current_tags.push_back(cv.substr(
					current_category.size() + 1));
				save_values = true;
			}
			next_token();
			while (current_token == T_TAG) {
				size_t clen = current_category.size();
				cv = current_value();
				string category = cv.substr(0, clen);
				if (category != current_category
				|| (DDL_v2 && cv[clen] != '.')
				|| (!DDL_v2 && cv[clen] != '_'))
					throw error("loop_ may only be for one category");
				if (keep)
					current_tags.push_back(
							cv.substr(clen + 1));
				next_token();
			}
			if (save_values) {
				ParseCategory& pf = cii->second.func;
				seen.insert(current_category);
				first_row = true;
				pf(true);
			}
			if (one_table)
				return;
			// eat remaining values
			first_row = false;
			save_values = false;
			if (stylized_) {
				// if seen all tables, skip to next data_
				if (current_token == T_VALUE) {
					bool tags_okay = seen.size() < categories.size();
					stylized_next_keyword(tags_okay);
				}
			}
			else while (current_token == T_VALUE)
				next_token();
			continue;
		}
		case T_SOI:
			throw error("unexpected restart of input");
		case T_GLOBAL:
			if (stash.size() > 0)
				process_stash();
			global_block();
			next_token();
			continue;
		case T_SAVE:
			if (stash.size() > 0)
				process_stash();
			save_frame(current_value());
			next_token();
			continue;
		case T_STOP:
			throw error("unexpected stop_ keyword");
		case T_LEFT_BRACKET:
			throw error("unexpected left bracket");
		case T_RIGHT_BRACKET:
			throw error("unexpected right bracket");
		case T_TAG: {
			// collapse consectutive tag value pairs with the
			// same category
			values.reserve(60); // avoid default large alloc
			// TODO: CIF category tags (no . separator)
			current_category.clear();
			Categories::iterator cii = categories.end();
			string cv = current_value();
			size_t sep = cv.find('.');
			DDL_v2 = (sep != std::string::npos);
			for (;;) {
				string category;
				if (DDL_v2) {
					category = cv.substr(0, sep);
				} else {
					category = cv;
					sep = current_category.size();
					if (category.substr(0, sep) == current_category
					&& category[sep] == '_')
						category = current_category;
					else for (;;) {
						sep = category.rfind('_');
						if (sep == std::string::npos)
							break;
						category.resize(sep);
						if (categories.find(category)
							    != categories.end())
							break;
					}
				}
				if (current_category.empty()
				|| category != current_category) {
					const char* first_tag_pos = current_value_start;
					if (cii != categories.end()) {
						// flush current category
						save_values = false;
						ParseCategory& pf = cii->second.func;
						seen.insert(current_category);
						first_row = true;
						pf(false);
					}
					if (!current_category.empty()
					&& one_table)
						return;
					current_category = category;
					cii = categories.find(current_category);
					bool keep = cii != categories.end();
					if (keep && !one_table) {
						for (auto d: cii->second.dependencies) {
							if (seen.find(d) != seen.end())
								continue;
							keep = false;
							stash.emplace(current_category,
							      StashInfo(first_tag_pos, lineno));
							break;
						}
					}
					if (keep) {
						current_tags.push_back(cv.substr(
							current_category.size() + 1));
						save_values = true;
					} else
						save_values = false;
				} else if (save_values)
					current_tags.push_back(cv.substr(
						current_category.size() + 1));
				next_token();
				if (current_token != T_VALUE)
					throw error("expected data value after data name");
				if (save_values)
					values.push_back(current_value());
				next_token();
				if (current_token != T_TAG)
					break;
				string cv = current_value();
				if (DDL_v2)
					sep = cv.find('.');
			}
			if (cii != categories.end()) {
				// flush current category
				save_values = false;
				ParseCategory& pf = cii->second.func;
				seen.insert(current_category);
				first_row = true;
				pf(false);
				current_category.clear();
			}
			if (one_table)
				return;
			if (seen.size() == categories.size()
			&& (current_token < T_DATA || current_token > T_STOP))
				// if seen all tables, skip to next data_
				stylized_next_keyword(false);
			continue;
		}
		case T_VALUE:
			throw error("unexpected data value");
		case T_EOI:
			break;
		}
		break;	// double break
	}
	if (stash.size() > 0)
		process_stash();
}

void
CIFFile::reset_parse()
{
	// parsing state
	version_.clear();
	parsing = false;
	stylized_ = false;
	current_data_block.clear();
	current_category.clear();
	current_tags.clear();
	values.clear();
	first_row = false;
	columns.clear();
	seen.clear();
	stash.clear();

	// lexical state
	current_token = T_SOI;
	current_value_start = nullptr;
	current_value_end = nullptr;
	line = whole_file;
	lineno = 1;
	pos = line;
	save_values = false;
}

void
CIFFile::next_token()
{
	if (current_token == T_EOI)
		return;
	const char* e;		// one beyond end of current token
again:
	for (; *pos == ' ' || is_whitespace(*pos); ++pos) {
#ifdef CR_IS_EOL
		if (*pos == '\r') {
			if (*(pos + 1) == '\n')
				++pos;
			++lineno;
		} else
#endif
		if (*pos == '\n') {
			++lineno;
		}
	}
	switch (
#ifdef CASE_INSENSITIVE
		tolower(*pos)
#else
		*pos
#endif
	) {
	case '\0':
		current_token = T_EOI;
		return;
	case ';':
		// if (! (pos == line || (whole_file && is_eol(*(pos - 1)))) )
		if (pos != line && (!whole_file || is_not_eol(*(pos - 1))))
			goto data_value;
		// TODO: if ";\" then fold long lines
		if (current_data_block.empty())
			throw error("string outside of data block");
		++pos;
		if (save_values)
			current_value_tmp.clear();
		for (;;) {
			for (e = pos; is_not_eol(*e); ++e)
				continue;
			if (save_values)
				current_value_tmp += string(pos, e - pos);
			pos = e;
#ifdef CR_IS_EOL
			if (*pos == '\r' && *(pos + 1) == '\n')
				++pos;
#endif
			if (!*pos) {
				current_token = T_EOI;
				throw error("incomplete multiline data value");
			}
			++pos;
			++lineno;
			if (*pos == ';' && is_eol(*(pos + 1))) {
				if (*(pos + 1))
					++pos;
				break;
			}
			if (save_values)
				current_value_tmp += '\n';
		}
		current_token = T_VALUE;
		if (save_values) {
			current_value_start = current_value_tmp.c_str();
			current_value_end = current_value_start + current_value_tmp.size();
		}
		return;
	case '#':
		for (++pos; is_not_eol(*pos); ++pos)
			continue;
		goto again;
	case '_': {
		for (e = pos + 1; is_not_whitespace(*e); ++e)
			continue;
		current_value_tmp = string(pos + 1, e - pos - 1);
		current_value_start = current_value_tmp.c_str();
		current_value_end = current_value_start + current_value_tmp.size();
		current_token = T_TAG;
		pos = e;
		return;
	}
	case 'D':
	case 'd':
		for (e = pos + 1; is_not_whitespace(*e); ++e)
			continue;
		if (e - pos >= 5 && ICASEEQN_P1("data_", pos, 5)) {
			current_token = T_DATA;
			current_value_start = pos + 5;
			current_value_end = e;
			pos = e;
			return;
		}
		goto data_value_e_set;
	case 'G':
	case 'g':
		for (e = pos + 1; is_not_whitespace(*e); ++e)
			continue;
		if (e - pos == 7 && ICASEEQN_P1("global_", pos, 7)) {
			current_token = T_GLOBAL;
			pos = e;
			return;
		}
		goto data_value_e_set;
	case 'L':
	case 'l':
		for (e = pos + 1; is_not_whitespace(*e); ++e)
			continue;
		if (e - pos == 5 && ICASEEQN_P1("loop_", pos, e - pos)) {
			current_token = T_LOOP;
			pos = e;
			return;
		}
		goto data_value_e_set;
	case 'S':
	case 's':
		for (e = pos + 1; is_not_whitespace(*e); ++e)
			continue;
		if (e - pos >= 5 && ICASEEQN_P1("save_", pos, 5)) {
			current_token = T_SAVE;
			current_value_start = pos + 5;
			current_value_end = e;
			pos = e;
			return;
		}
		if (e - pos == 5 && ICASEEQN_P1("stop_", pos, 5)) {
			current_token = T_STOP;
			pos = e;
			return;
		}
		goto data_value_e_set;
	case '$':
		// save frame pointer -- treat like a value
		goto data_value;
	case '[':
		++pos;
		current_token = T_LEFT_BRACKET;
		return;
	case ']':
		++pos;
		current_token = T_RIGHT_BRACKET;
		return;
	case '"':
	case '\'': {
		char quote = *pos;
		e = pos + 1;
		for (;;) {
			for (; *e != quote; ++e) {
				if (is_eol(*e))
					throw error("unterminated string");
			}
			if (is_whitespace(*(e + 1)))
				break;
			++e;
		}
		current_token = T_VALUE;
		if (save_values) {
			current_value_start = pos + 1;
			current_value_end = e;
		}
		pos = e + 1;
		return;
	}
	default:
	data_value:
		for (e = pos + 1; is_not_whitespace(*e); ++e)
			continue;
	data_value_e_set:
		current_token = T_VALUE;
		if (save_values) {
			current_value_start = pos;
			current_value_end = e;
		}
		pos = e;
		return;
	}
	/* NOTREACHED */
}

void
CIFFile::stylized_next_keyword(bool tag_okay)
{
	// Search for keyword and optionally tags.
	// In a stylized PDBx/mmCIF file, all keywords are lowercase,
	// the tags are mixed case, and all are at the beginning of a line.
	for (;;) {
		for (; is_not_eol(*pos); ++pos)
			continue;
		if (!*pos) {
			current_token = T_EOI;
			return;
		}
		++pos;
		++lineno;
		switch (*pos) {
		case '\0':
			current_token = T_EOI;
			return;
		case ';':
			if (pos != line && (!whole_file || is_not_eol(*(pos - 1))))
				continue;
			++pos;
			for (;;) {
				for (; is_not_eol(*pos); ++pos)
					continue;
				if (!*pos) {
					current_token = T_EOI;
					throw error("incomplete multiline data value"); }
				++pos;
				++lineno;
				if (*pos == ';' && is_eol(*(pos + 1))) {
					if (*(pos + 1))
						++pos;
					break;
				}
			}
			continue;
		case '_': {
			if (not tag_okay)
				continue;
			const char* e;
			for (e = pos + 1; is_not_whitespace(*e); ++e)
				continue;
			current_value_tmp = string(pos + 1, e - pos - 1);
			current_value_start = current_value_tmp.c_str();
			current_value_end = current_value_start + current_value_tmp.size();
			current_token = T_TAG;
			pos = e;
			return;
		}
		case 'd':
			if (STRNEQ_P1("data_", pos, 5)) {
				current_token = T_DATA;
				current_value_start = pos + 5;
				for (current_value_end = current_value_start;
				     is_not_whitespace(*current_value_end);
				     ++current_value_end)
					continue;
				pos = current_value_end;
				return;
			}
			continue;
		case 'g':
			if (STRNEQ_P1("global_", pos, 7)) {
				current_token = T_GLOBAL;
				pos += 7;
				return;
			}
			continue;
		case 'l':
			if (STRNEQ_P1("loop_", pos, 5)
			&& is_whitespace(*(pos + 5))) {
				current_token = T_LOOP;
				pos += 5;
				return;
			}
			continue;
		case 's':
			if (STRNEQ_P1("save_", pos, 5)) {
				current_token = T_SAVE;
				current_value_start = pos + 5;
				for (current_value_end = current_value_start;
				     is_not_whitespace(*current_value_end);
				     ++current_value_end)
					continue;
				pos = current_value_end;
				return;
			}
			if (STRNEQ_P1("stop_", pos, 5)
			&& is_whitespace(*(pos + 5))) {
				current_token = T_STOP;
				pos += 5;
				return;
			}
			continue;
		default:
			continue;
		}
	}
}

std::vector<int>
CIFFile::find_column_offsets()
{
	// Find starting character position of each table column on a line
	std::vector<int> offsets;
	const char* save_start = current_value_start;
	const char* start = save_start;
	if (is_not_whitespace(*(start - 1)))
		--start;	// must have had a leading quote
	if (is_not_eol(*(start - 1)))
		return offsets;	// isn't at start of line, so not stylized
	int size = current_tags.size();
	offsets.reserve(size + 1);	// save one extra for end of line
	offsets.push_back(0);	// first column starts at beginning of line
	const char* save_pos = pos;
	size_t save_lineno = lineno;
	for (int i = 1; i != size; ++i) {
		next_token();
		if (is_whitespace(*(current_value_start - 1)))
			offsets.push_back(current_value_start - start);
		else {
			// must have had a leading quote
			offsets.push_back(current_value_start - start - 1);
		}
	}
	if (lineno != save_lineno) {
		// Values were not all on the same line,
		// so fallback to tokenizing.
		offsets.clear();
		pos = save_pos;
		lineno = save_lineno;
	} else {
#ifdef FIXED_LENGTH_ROWS
		// all rows are the same length (padded with trailing spaces)
		while (is_not_eol(*current_value_end))
			++current_value_end;
		offsets.push_back(current_value_end - start);
#else
		// The extra slot to be filled in later -- if stylized lines are
		// fixed length, then this could be filled in now.
		offsets.push_back(0);
#endif
		current_value_start = save_start;
	}
	return offsets;
}

int
CIFFile::get_column(const char *tag, bool required)
{
	auto i = std::find(current_tags.begin(), current_tags.end(), std::string(tag));
	if (i != current_tags.end())
		return i - current_tags.begin();
	if (!required)
		return -1;
	std::ostringstream os;
	os << "Missing tag " << tag << " in category " << current_category;
	throw error(os.str());
}

bool
CIFFile::parse_row(ParseValues& pv)
{
	if (current_category.empty())
		// not category or exhausted values
		throw error("no values available");
	if (current_tags.empty())
		return false;
	if (first_row) {
		first_row = false;
		columns.clear();
		std::sort(pv.begin(), pv.end(), 
			[](const ParseColumn& a, const ParseColumn& b) -> bool {
				return a.column < b.column;
			});
		if (stylized_ && values.empty())
			columns = find_column_offsets();
	}
	auto pvi = pv.begin(), pve = pv.end();
	while (pvi != pve && pvi->column < 0)
		++pvi;
	if (!values.empty()) {
		// values were given per-tag
		// assert(current_tags.size() == values.size())
		for (; pvi != pve; ++pvi) {
			const char *buf = values[pvi->column].c_str();
			current_value_start = buf;
			current_value_end = buf + values[pvi->column].size();
			pvi->func(current_value_start, current_value_end);
		}
		current_tags.clear();
		values.clear();
		return true;
	}
	if (current_token != T_VALUE) {
		current_tags.clear();
		return false;
	}
	if (pvi == pve) {
		// discard row
		if (!columns.empty()) {
#ifdef FIXED_LENGTH_ROWS
			pos += columns[columns.size() - 1] + 1;
			++lineno;
#else
			if (columns.size() > 2) {
				// jump to last column -- if lines were padded
				// with trailing space, then could jump to end
				pos += columns[columns.size() - 2];
			}
			for (; is_not_eol(*pos); ++pos)
				continue;
#endif
#ifdef COMMENT_TERMINATED
			if (*pos == '#')
				next_token();
			else
				current_value_start = pos;
#else
			next_token();
#endif
			return true;
		}
		for (int i = 0, e = current_tags.size(); i < e; ++i) {
			if (current_token != T_VALUE)
				throw error("not enough data values");
			next_token();
		}
		return true;
	}
	if (!columns.empty()) {
		// stylized parsing
		const char* start = current_value_start;
		if (is_not_whitespace(*(start - 1)))
			--start;	// must have had a leading quote
		if (is_not_eol(*(start - 1))) {
			// isn't at start of line, so not stylized
			throw error("PDBx/mmCIF styling lost");
		}
#ifndef FIXED_LENGTH_ROWS
		// rows are not padded with trailing spaces
		if (columns.size() > 2) {
			// jump to last column
			pos = start + columns[columns.size() - 2];
		}
		for (; is_not_eol(*pos); ++pos)
			continue;
		columns.back() = pos - start;
#endif
		for (; pvi != pve; ++pvi) {
			current_value_start = start + columns[pvi->column];
			current_value_end = start + columns[pvi->column + 1];
			if (*current_value_start == '\''
			|| *current_value_start == '"') {
				// strip leading and trailng quotes
				--current_value_end;
				while (*current_value_end != *current_value_start)
					--current_value_end;
				++current_value_start;
			} else if (pvi->need_end) {
				// strip trailing whitespace
				--current_value_end;
				while (*current_value_end == ' '
				|| is_whitespace(*current_value_end))
					--current_value_end;
				++current_value_end;
			}
			pvi->func(current_value_start, current_value_end);
		}
#ifdef FIXED_LENGTH_ROWS
		pos = start + columns[columns.size() - 1] + 1;
		++lineno;
#endif
#ifdef COMMENT_TERMINATED
		if (*pos == '#')
			next_token();
		else
			current_value_start = pos;
#else
		next_token();
#endif
		return true;
	}
	for (int i = 0, e = current_tags.size(); i < e; ++i) {
		if (current_token != T_VALUE)
			throw error("not enough data values");
		if (i == pvi->column) {
			pvi->func(current_value_start, current_value_end);
			++pvi;
			if (pvi == pve) {
				// make (i == pvi->column) false
				pvi = pv.begin();
			}
		}
		next_token();
	}
	return true;
}

void
CIFFile::process_stash()
{
	Token last_token = current_token;
	size_t last_lineno = lineno;
	current_token = T_SOI;	// make sure next_token return values
	auto save_stash = std::move(stash);
	stash.clear();
	for (auto c: categoryOrder) {
		if (seen.find(c) != seen.end())
			continue;
		auto si = save_stash.find(c);
		if (si == save_stash.end())
			throw error("missing category: " + c);
		pos = si->second.start;
		lineno = si->second.lineno;
		internal_parse(true);
	}
	current_token = last_token;
	lineno = last_lineno;
}

void
CIFFile::data_block(const string& /*name*/)
{
	// By default, ignore data block declarations
}

void
CIFFile::save_frame(const string& /*code*/)
{
	throw error("unexpected save_ keyword");
}

void
CIFFile::global_block()
{
	throw error("unexpected global_ keyword");
}

} // namespace readcif
