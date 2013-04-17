#import pdbmtx_atoms as data
#import pdb3fx2_atoms as data
#import pdb3k9f_atoms as data
import pdb3cc4_atoms as data

hist = {}
for item in data.data:
    if item[0] == 's':
        # sphere: 's', radius, [x, y, z], [r, g, b, a]
        radius = item[1]
        count = hist.get(radius, 0)
        hist[radius] = count + 1

range = list(hist.keys())
range.sort()
for v in range:
    print v, hist[v]
