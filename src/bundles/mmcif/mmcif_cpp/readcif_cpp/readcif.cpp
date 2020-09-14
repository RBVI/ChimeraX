// vi: set noexpandtab ts=8 sw=8:
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
 *      Institutes of Health R01-GM129325.
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

#define CR_IS_EOL	/* undef for ~2% speedup */
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
#ifdef _WIN32
# define UNICODE
# include <windows.h>
#else
# include <fcntl.h>
# include <unistd.h>
# include <sys/mman.h>
# include <sys/stat.h>
# include <errno.h>
#endif

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

inline int
is_eol(char c)
{
#ifdef CR_IS_EOL
	return c == 0 || c == '\n' || c == '\r';
#else
	return c == 0 || c == '\n';
#endif
}

inline int
is_not_eol(char c)
{
#ifdef CR_IS_EOL
	return c && c != '\n' && c != '\r';
#else
	return c && c != '\n';
#endif
}

#define STRNEQ_P1(name, buf, len) (strncmp((name) + 1, (buf) + 1, (len) - 1) == 0)
#ifndef CASE_INSENSITIVE
#define ICASEEQN_P1(name, buf, len) STRNEQ_P1(name, buf, len)
#else
// icaseeqn:
//	compare name in a case independent way to buf
inline bool
icaseeqn(const char* name, const char* buf, size_t len)
{
	// This only works for ASCII characters
	for (size_t i = 0; i < len; ++i) {
		if (name[i] == '\0' || buf[i] == '\0')
			return name[i] == buf[i];
		if (tolower(name[i]) != tolower(buf[i]))
			return false;
	}
	return true;
}

inline bool
icaseeqn(const string& s0, const string& s1)
{
	if (s0.size() != s1.size())
		return false;
	return icaseeqn(s0.c_str(), s1.c_str());
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
	internal_reset_parse();
}

const char* CIFFile::token_names[] = {
	"Start of intput",	// T_SOI
	"data_ block",		// T_DATA
	"global_ keyword",	// T_GLOBAL
	"loop_ keyword", 	// T_LOOP
	"save_ frame",		// T_SAVE
	"stop_ keyword",	// T_STOP
	"data name",		// T_TAG
	"data value",		// T_VALUE
	"unquoted left square bracket",	// T_LEFT_SQUARE_BRACKET
	"unquoted right square brecket",	// T_RIGHT_SQUARE_BRACKET
	"End of input"		// T_EOI
};

void
CIFFile::register_category(const string& category, ParseCategory callback,
					const StringVector& dependencies)
{
#ifdef CASE_INSENSITIVE
	string cname = category;
	for (auto& c: cname)
		c = tolower(c);
	StringVector deps = dependencies;
	for (auto& dep: deps) {
		for (auto& c: dep)
			c = tolower(c);
	}
#else
	const string& cname = category;
	const StringVector& deps = dependencies;
#endif

	for (auto& dep: dependencies) {
		if (categories.find(dep) != categories.end())
			continue;
		std::ostringstream err_msg;
		err_msg << "Reference to unregistered dependency '" << dep
			<< "' in category '" << category << "'";
		throw std::logic_error(err_msg.str());
	}
	if (callback) {
		categoryOrder.push_back(category);
		categories.emplace(cname,
			   CategoryInfo(category, callback, deps));
	} else {
		// TODO: find category in categoryOrder
		// make sure none of the later categories depend on it
		throw std::runtime_error("missing category callback");
		categories.erase(category);
	}
}

void
CIFFile::set_PDBx_fixed_width_columns(const std::string& category)
{
#ifdef CASE_INSENSITIVE
	string c(category);
	for (auto& c: current_category)
		c = tolower(c);
#else
	const string& c = category;
#endif
	use_fixed_width_columns.insert(c);
}

#ifdef _WIN32
void
throw_windows_error(DWORD err_num, const char* where)
{
	wchar_t *message_buffer;
	FormatMessageW(FORMAT_MESSAGE_ALLOCATE_BUFFER
			| FORMAT_MESSAGE_FROM_SYSTEM
			| FORMAT_MESSAGE_IGNORE_INSERTS,
		NULL, err_num,
		MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
		(wchar_t*) &message_buffer, 0, NULL);
	int len = WideCharToMultiByte(CP_UTF8, 0, message_buffer, -1,
					NULL, 0, 0, 0);
	std::string message;
	message.resize(len);
	(void) WideCharToMultiByte(CP_UTF8, 0, message_buffer, -1,
				&message[0], len, 0, 0);

	std::ostringstream err_msg;
	if (where)
		err_msg << where << ": ";
	err_msg << message;
	HeapFree(GetProcessHeap(), 0, message_buffer);
	throw std::runtime_error(err_msg.str());
}
#endif

void
CIFFile::parse_file(const char* filename)
{
	std::ostringstream err_msg;
#ifdef _WIN32
	size_t len = strlen(filename);
	std::vector<wchar_t> wfilename(len + 1);
	MultiByteToWideChar(CP_UTF8, 0, filename, len, &wfilename[0], len + 1);
	HANDLE file = CreateFileW(&wfilename[0], GENERIC_READ, FILE_SHARE_READ, NULL,
			OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
	if (file == INVALID_HANDLE_VALUE) {
		DWORD err_num = GetLastError();
		throw_windows_error(err_num, "opening file for reading");
	}
	LARGE_INTEGER size;
	if (!GetFileSizeEx(file, &size)) {
		DWORD err_num = GetLastError();
		CloseHandle(file);
		throw_windows_error(err_num, "getting file size");
	}
	HANDLE mapping = CreateFileMapping(file, NULL, PAGE_READONLY, 0, 0, NULL);
	if (mapping == INVALID_HANDLE_VALUE) {
		DWORD err_num = GetLastError();
		CloseHandle(file);
		throw_windows_error(err_num, "creating file mapping");
	}
	void *buffer = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, size.QuadPart /*+ 1*/);
	if (buffer == NULL) {
		DWORD err_num = GetLastError();
		CloseHandle(file);
		CloseHandle(mapping);
		throw_windows_error(err_num, "creating file view");
	}
	try {
		parse(reinterpret_cast<char*>(buffer));
	} catch (...) {
		UnmapViewOfFile(buffer);
		CloseHandle(mapping);
		CloseHandle(file);
		throw;
	}
	UnmapViewOfFile(buffer);
	CloseHandle(mapping);
	CloseHandle(file);
#else
	int fd = open(filename, O_RDONLY);
	if (fd == -1) {
		int err_num = errno;
		err_msg << "open: " << strerror(err_num);
		throw std::runtime_error(err_msg.str());
	}
	struct stat sb;
	if (fstat(fd, &sb) == -1) {
		int err_num = errno;
		err_msg << "stat: " << strerror(err_num);
		throw std::runtime_error(err_msg.str());
	}

	bool used_mmap = false;
	char* buffer = NULL;

	long page_size = sysconf(_SC_PAGESIZE);
	if (sb.st_size % page_size != 0) {
		void *buf = mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE,
				fd, 0);
		if (buf == MAP_FAILED) {
			int err_num = errno;
			err_msg << "mmap: " << strerror(err_num);
			throw std::runtime_error(err_msg.str());
		}
		buffer = reinterpret_cast<char*>(buf);
		used_mmap = true;
	} else {
		buffer = new char [sb.st_size + 1];
		if (read(fd, buffer, sb.st_size) == -1) {
			int err_num = errno;
			err_msg << "read: " << strerror(err_num);
			throw std::runtime_error(err_msg.str());
		}
		buffer[sb.st_size] = '\0';
	}
	try {
		parse(reinterpret_cast<const char*>(buffer));
	} catch (...) {
		(void) close(fd);
		if (used_mmap)
			(void) munmap(buffer, sb.st_size + 1);
		else
			delete [] buffer;
		throw;
	}
	(void) close(fd);
	if (used_mmap) {
		if (munmap(buffer, sb.st_size + 1) == -1) {
			int err_num = errno;
			err_msg << "munmap: " << strerror(err_num);
			throw std::runtime_error(err_msg.str());
		}
	} else {
		delete [] buffer;
	}
#endif
}

void
CIFFile::parse(const char* buffer)
{
	whole_file = buffer;
	try {
		if (parsing)
			throw error("Already parsing");
		internal_reset_parse();
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
#ifdef CR_IS_EOL
			if (*pos == '\r')
				++pos;
#endif
		}
		internal_parse();
		parsing = false;
		finished_parse();
	} catch (std::exception &e) {
		parsing = false;
		throw;
	}
}

std::runtime_error
CIFFile::error(const string& text, size_t lineno)
{
	if (lineno == 0)
		lineno = this->lineno;
	std::ostringstream err_msg;
	err_msg << text << " near line " << lineno;
	return std::move(std::runtime_error(err_msg.str()));
}

inline string
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
			current_data_block = current_value();
			data_block(current_data_block);
			if (stylized)
				stylized_next_keyword(true);
			else
				next_token();
			continue;
		case T_LOOP: {
			const char* loop_pos = pos - 5;
			next_token();
			if (current_token != T_TAG) {
				std::ostringstream err_msg;
				err_msg << "expected data name after loop_, not "
					<< token_names[current_token];
				throw error(err_msg.str());
			}
			Categories::iterator cii;
			string cv = current_value();
			size_t sep = cv.find('.');
			DDL_v2 = (sep != string::npos);
			if (DDL_v2) {
#ifndef CASE_INSENSITIVE
				current_category = cv.substr(0, sep);
#else
				current_category_cp = cv.substr(0, sep);
				current_category = current_category_cp;
				for (auto& c: current_category)
					c = tolower(c);
#endif
				cii = categories.find(current_category);
			} else {
				cii = categories.end();
#ifndef CASE_INSENSITIVE
				current_category = cv;
#else
				current_category_cp = cv;
				current_category = current_category_cp;
				for (auto& c: current_category)
					c = tolower(c);
#endif
				for (;;) {
					sep = current_category.rfind('_');
					if (sep == string::npos)
						break;
					current_category.resize(sep);
#ifdef CASE_INSENSITIVE
					current_category_cp.resize(sep);
#endif
					cii = categories.find(current_category);
					if (cii != categories.end()) {
						// if already seen, then
						// category is a prefix
						if (seen.find(current_category)
								!= seen.end())
							cii = categories.end();
						break;
					}
				}
			}
			save_values = cii != categories.end();
			if (!save_values && unregistered)
				save_values = true;
			else if (save_values && !one_table) {
				for (auto d: cii->second.dependencies) {
					if (seen.find(d) != seen.end())
						continue;
					save_values = false;
					stash.emplace(current_category,
						  StashInfo(loop_pos, lineno));
					break;
				}
			}
			if (save_values) {
				string colname(cv.substr(
					current_category.size() + 1));
#ifdef CASE_INSENSITIVE
				current_colnames_cp.push_back(colname);
				for (auto& c: colname)
					c = tolower(c);
#endif
				current_colnames.emplace_back(colname);
			}
			next_token();
			while (current_token == T_TAG) {
				size_t clen = current_category.size();
				cv = current_value();
				string category = cv.substr(0, clen);
				if (
#ifdef CASE_INSENSITIVE
				    category != current_category_cp
#else
				    category != current_category
#endif
				|| (DDL_v2 && cv[clen] != '.')
				|| (!DDL_v2 && cv[clen] != '_'))
					throw error("loop_ may only be for one category");
				if (save_values) {
					string colname(cv.substr(clen + 1));
#ifdef CASE_INSENSITIVE
					current_colnames_cp.push_back(colname);
					for (auto& c: colname)
						c = tolower(c);
#endif
					current_colnames.emplace_back(colname);
				}
				next_token();
			}
			if (save_values) {
				seen.insert(current_category);
				first_row = true;
				in_loop = true;
				ParseCategory& pf = (cii != categories.end()) ? cii->second.func : unregistered;
				pf();
				first_row = false;
				current_category.clear();
				current_colnames.clear();
				current_colnames_cp.clear();
				values.clear();
				save_values = false;
			}
			if (one_table)
				return;
			// eat remaining values
			if (stylized) {
				// if seen all tables, skip to next keyword
				bool tags_okay = seen.size() < categories.size();
				if (!current_is_keyword()
				&& !(tags_okay && current_token == T_TAG))
					stylized_next_keyword(tags_okay);
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
		case T_LEFT_SQUARE_BRACKET:
			// CIF 1.1 standard reserves left square bracket for future use
			throw error("left square bracket is illegal outside of quotes in CIF 1.1");
		case T_RIGHT_SQUARE_BRACKET:
			// CIF 1.1 standard reserves right square bracket for future use
			throw error("right square bracket is illegal outside of quotes in CIF 1.1");
		case T_TAG: {
			// collapse consectutive tag value pairs with the
			// same category
			values.reserve(60); // avoid default large alloc
			// TODO: CIF category tags (no . separator)
			current_category.clear();
			Categories::iterator cii = categories.end();
			string cv = current_value();
			size_t sep = cv.find('.');
			DDL_v2 = (sep != string::npos);
			for (;;) {
				string category;
#ifdef CASE_INSENSITIVE
				string category_cp;
#endif
				if (DDL_v2) {
					category = cv.substr(0, sep);
#ifdef CASE_INSENSITIVE
					category_cp = category;
					for (auto& c: category)
						c = tolower(c);
#endif
				} else {
					category = cv;
#ifdef CASE_INSENSITIVE
					category_cp = category;
					for (auto& c: category)
						c = tolower(c);
#endif
					sep = current_category.size();
					if (category.substr(0, sep) == current_category
					&& category[sep] == '_') {
						category = current_category;
#ifdef CASE_INSENSITIVE
						category_cp = current_category_cp;
#endif
					} else for (;;) {
						sep = category.rfind('_');
						if (sep == string::npos)
							break;
						category.resize(sep);
#ifdef CASE_INSENSITIVE
						category_cp.resize(sep);
#endif
						if (categories.find(category)
								!= categories.end()) {
							// if already seen, then
							// category is a prefix
							if (seen.find(current_category)
									!= seen.end())
								cii = categories.end();
							break;
						}
					}
				}
				if (current_category.empty()
				|| category != current_category) {
					const char* first_tag_pos = pos - cv.size() - 1;
					if (save_values) {
						// flush current category
						seen.insert(current_category);
						first_row = true;
						in_loop = false;
						ParseCategory& pf = (cii != categories.end()) ? cii->second.func : unregistered;
						pf();
						first_row = false;
						//current_category.clear();
						current_colnames.clear();
						current_colnames_cp.clear();
						values.clear();
						save_values = false;
					}
					if (!current_category.empty()
					&& one_table) {
						current_category.clear();
						return;
					}
					current_category = category;
#ifdef CASE_INSENSITIVE
					current_category_cp = category_cp;
#endif
					cii = categories.find(current_category);
					save_values = cii != categories.end();
					if (!save_values && unregistered)
						save_values = true;
					else if (save_values && !one_table) {
						for (auto d: cii->second.dependencies) {
							if (seen.find(d) != seen.end())
								continue;
							save_values = false;
							stash.emplace(current_category,
								  StashInfo(first_tag_pos, lineno));
							break;
						}
					}
					if (save_values) {
						string colname(cv.substr(
							current_category.size() + 1));
#ifdef CASE_INSENSITIVE
						current_colnames_cp.push_back(colname);
						for (auto& c: colname)
							c = tolower(c);
#endif
						current_colnames.emplace_back(colname);
					}
				} else if (save_values) {
					string colname(cv.substr(
						current_category.size() + 1));
#ifdef CASE_INSENSITIVE
					current_colnames_cp.push_back(colname);
					for (auto& c: colname)
						c = tolower(c);
#endif
					current_colnames.emplace_back(colname);
				}
				next_token();
				if (current_token != T_VALUE) {
					std::ostringstream err_msg;
					err_msg << "expected data value after data name, not "
						<< token_names[current_token];
					throw error(err_msg.str());
				}
				if (save_values)
					//values.push_back(current_value());
					values.emplace_back(current_value_start,
									current_value_end - current_value_start);
				next_token();
				if (current_token != T_TAG)
					break;
				cv = current_value();
				if (DDL_v2)
					sep = cv.find('.');
			}
			if (save_values) {
				// flush current category
				seen.insert(current_category);
				first_row = true;
				in_loop = false;
				ParseCategory& pf = (cii != categories.end()) ? cii->second.func : unregistered;
				pf();
				first_row = false;
				current_category.clear();
				current_colnames.clear();
				current_colnames_cp.clear();
				values.clear();
				save_values = false;
			}
			if (one_table)
				return;
			if (seen.size() == categories.size()
			&& !current_is_keyword())
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
CIFFile::internal_reset_parse()
{
	// parsing state
	version_.clear();
	parsing = false;
	stylized = false;
	use_fixed_width_columns.clear();
	current_data_block.clear();
	current_category.clear();
	current_colnames.clear();
	current_colnames_cp.clear();
	values.clear();
	in_loop = false;
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
	reset_parse();
}

void
CIFFile::reset_parse()
{
}

void
CIFFile::finished_parse()
{
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
	switch (*pos) {
	case '\0':
		current_token = T_EOI;
		return;
	case ';': {
		size_t start_lineno = lineno;
		// if (! (pos == line || (whole_file && is_eol(*(pos - 1)))) )
		if (pos != line && (!whole_file || is_not_eol(*(pos - 1))))
			goto data_value;
		// TODO: if ";\" then fold long lines
		if (current_data_block.empty())
			throw error("string outside of data block", start_lineno);
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
				throw error("incomplete multiline data value", start_lineno);
			}
			++pos;
			++lineno;
			if (*pos == ';') {
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
	}
	case '#':
		for (++pos; is_not_eol(*pos); ++pos)
			continue;
#ifdef CR_IS_EOL
		if (*pos == '\r')
			++pos;
#endif
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
#ifdef CASE_INSENSITIVE
	case 'D':
#endif
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
#ifdef CASE_INSENSITIVE
	case 'G':
#endif
	case 'g':
		for (e = pos + 1; is_not_whitespace(*e); ++e)
			continue;
		if (e - pos == 7 && ICASEEQN_P1("global_", pos, 7)) {
			current_token = T_GLOBAL;
			pos = e;
			return;
		}
		goto data_value_e_set;
#ifdef CASE_INSENSITIVE
	case 'L':
#endif
	case 'l':
		for (e = pos + 1; is_not_whitespace(*e); ++e)
			continue;
		if (e - pos == 5 && ICASEEQN_P1("loop_", pos, e - pos)) {
			current_token = T_LOOP;
			pos = e;
			return;
		}
		goto data_value_e_set;
#ifdef CASE_INSENSITIVE
	case 'S':
#endif
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
		current_token = T_LEFT_SQUARE_BRACKET;
		return;
	case ']':
		++pos;
		current_token = T_RIGHT_SQUARE_BRACKET;
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
		case ';': {
			size_t start_lineno = lineno;
			if (pos != line && (!whole_file || is_not_eol(*(pos - 1))))
				continue;
			++pos;
			for (;;) {
				for (; is_not_eol(*pos); ++pos)
					continue;
#ifdef CR_IS_EOL
				if (*pos == '\r' && *(pos + 1) == '\n')
					++pos;
#endif
				if (!*pos) {
					current_token = T_EOI;
					throw error("incomplete multiline data value", start_lineno);
				}
				++pos;
				++lineno;
				char c = *(pos + 1);
				if (*pos == ';' && is_eol(c)) {
#ifdef CR_IS_EOL
					if (c == '\r') {
						++pos;
						c = *(pos + 1);
					}
#endif
					if (c)
						++pos;
					break;
				}
			}
			continue;
		}
		case '_': {
			if (!tag_okay)
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

vector<int>
CIFFile::find_column_offsets()
{
	// Find starting character position of each table column on a line
	vector<int> offsets;
	const char* save_start = current_value_start;
	const char* start = save_start;
	if (is_not_whitespace(*(start - 1)))
		--start;	// must have had a leading quote
	if (is_not_eol(*(start - 1)))
		return offsets;	// isn't at start of line, so not stylized
	int size = current_colnames.size();
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
		if (*current_value_end == '\r' && *(current_value_end + 1) == '\n')
			++current_value_end;	// check for DOS line ending
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
CIFFile::get_column(const char* name, bool required)
{
	if (current_colnames.empty())
		throw std::runtime_error("must be parsing a table before getting a column position");
	string colname(name);
#ifdef CASE_INSENSITIVE
	for (auto& c: colname)
		c = tolower(c);
#endif
	auto i = std::find(current_colnames.begin(), current_colnames.end(), colname);
	if (i != current_colnames.end())
		return i - current_colnames.begin();
	if (!required)
		return -1;
	std::ostringstream err_msg;
	err_msg << "Missing column '" << name << "'" /*<< " in category " << current_category*/;
	throw error(err_msg.str());
}

bool
CIFFile::parse_row(ParseValues& pv)
{
	if (current_category.empty())
		// not category or exhausted values
		throw error("no values available");
	if (current_colnames.empty())
		return false;
	if (first_row) {
		first_row = false;
		bool fixed = use_fixed_width_columns.count(current_category);
		columns.clear();
		std::sort(pv.begin(), pv.end(),
			[](const ParseColumn& a, const ParseColumn& b) -> bool {
				return a.column < b.column;
			});
		if (fixed && stylized && values.empty())
			columns = find_column_offsets();
	}
	auto pvi = pv.begin(), pve = pv.end();
	while (pvi != pve && pvi->column < 0)
		++pvi;
	if (!values.empty()) {
		// values were given per-tag
		// assert(current_colnames.size() == values.size())
		for (; pvi != pve; ++pvi) {
			const char* buf = values[pvi->column].c_str();
			current_value_start = buf;
			if (pvi->need_end) {
				current_value_end = buf + values[pvi->column].size();
				pvi->func2(current_value_start, current_value_end);
			} else
				pvi->func1(current_value_start);
		}
		current_colnames.clear();
		current_colnames_cp.clear();
		values.clear();
		return true;
	}
	if (current_token != T_VALUE) {
		current_colnames.clear();
		current_colnames_cp.clear();
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
		for (int i = 0, e = current_colnames.size(); i < e; ++i) {
			if (current_token != T_VALUE) {
				std::ostringstream err_msg;
				err_msg << "not enough data values, found "
					<< token_names[current_token];
				throw error(err_msg.str());
			}
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
			if (pvi->need_end)
				pvi->func2(current_value_start, current_value_end);
			else
				pvi->func1(current_value_start);
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
	for (int i = 0, e = current_colnames.size(); i < e; ++i) {
		if (current_token != T_VALUE) {
			std::ostringstream err_msg;
			err_msg << "not enough data values, found"
				<< token_names[current_token];
			throw error(err_msg.str());
		}
		if (i == pvi->column) {
			if (pvi->need_end)
				pvi->func2(current_value_start, current_value_end);
			else
				pvi->func1(current_value_start);
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

StringVector&
CIFFile::parse_whole_category()
{
	if (current_category.empty())
		// not category or exhausted values
		throw error("no values available");
	if (current_colnames.empty())
		return values;

	if (!values.empty()) {
		// values were given per-tag
		// assert(current_colnames.size() == values.size())
		current_colnames.clear();
		current_colnames_cp.clear();
		return values;
	}

	//values.reserve(current_colnames.size());
	values.reserve(4000000);
	while (current_token == T_VALUE) {
		//values.push_back(current_value());
		values.emplace_back(current_value_start,
				  current_value_end - current_value_start);
		next_token();
	}
	// assert(data.size() % current_colnames.size() == 0);

	current_colnames.clear();
	current_colnames_cp.clear();
	return values;
}

void
CIFFile::parse_whole_category(ParseValue2 func)
{
	if (current_category.empty())
		// not category or exhausted values
		throw error("no values available");
	if (current_colnames.empty())
		return;


	if (!values.empty()) {
		// values were given per-tag
		// assert(current_colnames.size() == values.size())
		for (auto& s: values) {
			const char* start = &s[0];
			const char* end = start + s.size();
			func(start, end);
		}
		current_colnames.clear();
		current_colnames_cp.clear();
		return;
	}

	while (current_token == T_VALUE) {
		func(current_value_start, current_value_end);
		next_token();
	}
	current_colnames.clear();
	current_colnames_cp.clear();
}

void
CIFFile::process_stash()
{
	const char* last_pos = pos;
	Token last_token = current_token;
	const char* last_value_start = current_value_start;
	const char* last_value_end = current_value_end;
	std::string last_value_tmp = current_value_tmp;
	size_t last_lineno = lineno;
	auto save_stash = std::move(stash);
	stash.clear();
	for (auto c: categoryOrder) {
		if (seen.find(c) != seen.end())
			continue;
		auto si = save_stash.find(c);
		if (si == save_stash.end()) {
			//std::cerr << error("missing category: " + c).what() << '\n';
			continue;
		}
		pos = si->second.start;
		lineno = si->second.lineno;
		current_token = T_SOI;	// make sure next_token returns values
		internal_parse(true);
	}
	pos = last_pos;
	current_token = last_token;
	current_value_start = last_value_start;
	current_value_end = last_value_end;
	current_value_tmp = last_value_tmp;
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
