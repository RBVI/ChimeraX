// vi: set expandtab shiftwidth=4 softtabstop=4:
//
// Output the number of rows and fields for all categories in a CIF file
//
// counts file.cif

#include "readcif.h"
#include <iostream>

readcif::CIFFile extract;

void
counts()
{
    size_t num_fields = extract.tags().size();
    std::cout << extract.category() << ": " << num_fields << " field";
    if (num_fields > 1)
        std::cout << 's';
    readcif::CIFFile::ParseValues pv;
    unsigned long num_rows = 0;
    while (extract.parse_row(pv))
        ++num_rows;
    std::cout << ", " << num_rows << " row";
    if (num_rows > 1)
        std::cout << 's';
    std::cout << '\n';
}

int
main(int argc, char **argv)
{
    extract.set_unregistered_callback(counts);
    extract.parse_file(argv[1]);
}
