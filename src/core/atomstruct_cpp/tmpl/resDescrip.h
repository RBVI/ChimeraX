// vi: set expandtab ts=4 sw=4:
#ifndef templates_resDescrip
#define templates_resDescrip

namespace tmpl {

typedef struct resDescrip {
    const char *name, *descrip;
} ResDescript;

ResDescript res_descripts[] = {
    { "ALA", "alanine" },
    { "ARG", "arginine" },
    { "ASH", "protonated aspartic acid" },
    { "ASN", "asparagine" },
    { "ASP", "aspartic acid" },
    { "CYS", "cysteine" },
    { "CYX", "cysteine with disulfide bond" },
    { "GLH", "protonated glutamic acid" },
    { "GLN", "glutamine" },
    { "GLU", "glutamic acid" },
    { "GLY", "glycine" },
    { "HIS", "histidine" },
    { "HID", "delta-protonated histidine" },
    { "HIE", "epsilon-protonated histidine" },
    { "HIP", "double-protonated histidine" },
    { "HYP", "hydroxyproline" },
    { "ILE", "isoleucine" },
    { "LEU", "leucine" },
    { "LYS", "lysine" },
    { "MET", "methionine" },
    { "PHE", "phenylalanine" },
    { "PRO", "proline" },
    { "SER", "serine" },
    { "THR", "threonine" },
    { "TRP", "tryptophan" },
    { "TYR", "tyrosine" },
    { "VAL", "valine" }
};

}  // namespace tmpl

#endif  // templates_resDescrip
