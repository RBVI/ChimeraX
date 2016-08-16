// vi: set expandtab shiftwidth=4 softtabstop=4:
// The idea is to have an expr-like program that searchs mmCIF files
//
// cifgrep category.id1,id2 [cif-file]

#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>
#include <sysexits.h>
#include <iostream>
#include <vector>
#include <unordered_map>
#include "readcif.h"

using std::string;
using std::vector;
using std::unordered_map;
using readcif::CIFFile;

// global options
bool mmCIF_style;
bool show_data_block;
bool show_filename;
bool list_filename_only;
bool verbose;
vector<string> tags;

// globals
bool found_something;
const char* current_filename;
typedef unordered_map<string, string> CategoryInfo;
unordered_map<string, CategoryInfo> info;

struct TerminateEarly: std::exception
{
};

struct Extract: CIFFile
{
	Extract();
	virtual void reset_parse();
	virtual void finish_parse();
};

Extract::Extract()
{
}

void
Extract::reset_parse()
{
	if (mmCIF_style)
		set_PDB_style(true);
}

void
Extract::finish_parse()
{
}

Extract extract;

void
save_parse_info()
{
	const string& cat = extract.category();
	auto& cat_info = info[cat];

	CIFFile::ParseValues pv;
	pv.reserve(cat_info.size());
	for (auto& x: cat_info) {
		auto column = extract.get_column(x.first.c_str());
		if (column == -1) {
			if (verbose)
				std::cerr << "Missing " << cat << '.' << x.first << '\n';
			exit(EXIT_FAILURE);
		}
		pv.emplace_back(column, true,
			[&] (const char* start, const char* end) {
				cat_info[x.first] = string(start, end - start);
			}); }

	static bool shorten_tags = true;
	if (shorten_tags) {
		// modify tags to be just the column ids
		auto len = cat.size();
		for (auto& x: tags) {
			x = x.substr(len + 1);
		}
		shorten_tags = false;
	}
	while (extract.parse_row(pv)) {
		found_something = true;
		if (show_filename) {
			if (show_data_block)
				std::cout << extract.block_code();
			else
				std::cout << current_filename;
		}
		if (list_filename_only) {
			std::cout << '\n';
			throw TerminateEarly();
		}
                if (show_filename)
                    std::cout << ": ";
		bool rest = false;
		for (auto&& id: tags) {
			if (rest)
				std::cout << '\t';
			std::cout << cat_info[id];
			rest = true;
		}
		std::cout << '\n';
	}
};

void
parse_tags(char* text)
{
	char* tok;
	string category;

	bool ddl2 = strchr(text, '.') != NULL;
	if (!ddl2) {
		std::cerr << "tags must be category.id\n";
		exit(EX_USAGE);
	}
	while ((tok = strtok(text, ",")) != NULL) {
	       text = NULL;
	       if (*tok != '.') {
		       char *cp = strchr(tok, '.');
		       if (cp == NULL) {
			       std::cerr << "Ignoring '" << tok << "'\n";
			       continue;
		       }
		       category = string(tok, cp - tok);
		       string id = string(cp + 1);
		       auto& cat_info = info[category];
		       auto& x = cat_info[id];	// create as side-effect
		       tags.push_back(tok);
		       continue;
	       }
	       if (category.empty()) {
		       std::cerr << "Missing category for first tag\n";
		       exit(EX_USAGE);
	       }
	       string id = string(tok + 1);
	       auto& x = info[category][id];	// create as side-effect
	       tags.push_back(category + tok);
	}
	// TODO: support more than one category
	if (info.size() > 1) {
	       std::cerr << "So far, only one category is supported\n";
	       exit(EX_USAGE);
	}
}

int
main(int argc, char** argv)
{
	int opt;
        bool set_show_filename = false;

	while ((opt = getopt(argc, argv, "mdhHlv")) != -1) {
		switch (opt) {
			case 'm':
				mmCIF_style = true;
				break;
			case 'd':
				show_data_block = true;
				break;
			case 'h':
				show_filename = false;
                                set_show_filename = true;
				break;
			case 'H':
				show_filename = true;
                                set_show_filename = true;
				break;
			case 'l':
				list_filename_only = true;
				show_filename = true;
                                set_show_filename = true;
				break;
			case 'v':
				verbose = true;
				break;
			default: /* '?' */
				goto usage;
		}
	}

	if (optind + 1 >= argc) {
usage:
		std::cerr << "Usage: " << argv[0] <<
			" [-d] [-m] [-l] [-q] CIF_tags filename(s)\n"
			"\t-d\tShow data block instead of filename.\n"
                        "\t-h\tSuppress filename.\n"
                        "\t-H\tAlways show filename.\n"
			"\t-l\tIf a match is found, just list the filename.\n"
			"\t-m\tmmCIF style (lowercase keyword/tags at beginning of line).\n"
			"\tCIF tags are comma separated category.id values.\n"
			"\t\tOnly one category is supported.  Subsequent id's\n"
			"\t\tcan elide the category (.id is sufficient).\n"
			"\tFilenames are separated by whitespace.\n";
		exit(EX_USAGE);
	}

	parse_tags(argv[optind++]);

//std::cerr << "tags:\n";
//for (auto t: tags) std::cout << "  " << t << '\n';

	for (auto& x: info) {
		const string& category = x.first;
		extract.register_category(category, save_parse_info);
	}

        if (!set_show_filename)
            show_filename = (argc - optind) > 1;
	for (; optind < argc; ++optind) {
		current_filename = argv[optind];
		try {
			extract.parse_file(current_filename);
		} catch (TerminateEarly &) {
			continue;
		} catch (std::exception &e) {
			if (verbose)
				std::cerr << current_filename << ": " << e.what() << '\n';
		}
	}

	if (found_something)
		exit(EXIT_SUCCESS);
	exit(EXIT_FAILURE);
}
