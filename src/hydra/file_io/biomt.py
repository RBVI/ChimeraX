# -----------------------------------------------------------------------------
#
def pdb_biomt_matrices(pdb_text):

    lines = lines_with_prefix(pdb_text, 'REMARK 350')

    mtable = {}
    for line in lines:
        fields = line.split()
        if len(fields) >= 8 and fields[2].startswith('BIOMT'):
            try:
                matrix_num = int(fields[3])
            except ValueError:
                continue
            if not matrix_num in mtable:
                mtable[matrix_num] = [None, None, None]
            try:
                row = int(fields[2][5]) - 1
            except ValueError:
                continue
            if row >= 0 and row <= 2:
                try:
                    mtable[matrix_num][row] = tuple(float(x) for x in fields[4:8])
                except ValueError:
                    continue

    # Order matrices by matrix number.
    mordered = tuple(nm[1] for nm in sorted(mtable.items()) if not None in nm[1])
    from ..geometry.place import Place
    matrices = tuple(Place(m) for m in mordered)
    return matrices

# -----------------------------------------------------------------------------
#
def lines_with_prefix(text, prefix):

    lines = []
    s = 0
    while s < len(text):
        i = text.find(prefix,s)
        if i == -1:
            break
        ie = text.find('\n',i)
        if ie == -1:
            ie = len(text)
        lines.append(text[i:ie])
        s = ie
    return lines
