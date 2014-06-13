BEGIN {
	state = "unknown"
}
$1 ~ /^!.*XXX.unit.atoms$/ {
	state = "atoms"
	next
}
$1 ~ /^!.*XXX.unit.connect$/ {
	state = "connect"
	count = 1;
	next
}
$1 ~ /^!.*XXX.unit.connectivity$/ {
	state = "connectivity"
	next
}
$1 ~ /^!.*XXX.unit.positions$/ {
	state = "positions"
	count = 1;
	next
}
$1 ~ /^!/ {
	state = "unknown"
	next
}
{
	if (state == "unknown") {
		next
	} else if (state == "atoms") {
#!entry.XXX.unit.atoms table  str name  str type  int typex  int resx  int flags  int seq  int elmnt  dbl chg
# "N" "N" 0 1 131072 1 7 -0.4630000000
		atoms[$6] = "" $1 ", Element(" $7 ")"
		gsub("\"", "", $1)
		gsub("\\+", "p", $1)
		gsub("-", "m", $1)
		gsub("\\*", "_", $1)
		gsub("'", "q", $1)
		atomname[$6] = "atom_" $1
	} else if (state == "connect") {
#!entry.XXX.unit.connect array int
# 1
# 9
		link[count] = $1
		count += 1
	} else if (state == "connectivity") {
#!entry.XXX.unit.connectivity table  int atom1x  int atom2x  int flags
# 1 2 1
		connect[$1] = connect[$1] " " $2
	} else if (state == "positions") {
#!entry.XXX.unit.positions table  dbl x  dbl y  dbl z
# 3.3257702415 1.5479089590 -0.0000016073
		x[count] = $1
		y[count] = $2
		z[count] = $3
		count += 1
	}
}
END {
	print ""
	print "static Residue *"
	print "init_ZZZ(Molecule *m)"
	print "{"
	print "	Coord c;"
	print "	Residue *r = m->new_residue(\"YYY\");"
	for (i in atoms) {
		print "	Atom *" atomname[i] " = m->new_atom(" atoms[i] ");"
		print "	r->add_atom(" atomname[i] ");"
# coordinate template values
		print "	c.set_xyz(" x[i] "," y[i] "," z[i] ");"
		print "	" atomname[i] "->set_coord(c);"
	}
	if (link[1] != 0)
		print "	r->chief(" atomname[link[1]] ");"
	if (link[2] != 0)
		print "	r->link(" atomname[link[2]] ");"
	for (i in connect) {
#		(void) m->newBond(_CA, _CB);
		n = split(connect[i], c)
		j = 1
		while (j <= n) {
			# don't make the H-H bond in TIP3 water
			if (atomname[i] == "atom_H2" && atomname[c[j]] == "atom_H1") {
				j += 1
				continue
			}
			print "	(void) m->new_bond(" atomname[i] ", " atomname[c[j]] ");"
			j += 1
		}
	}
	print "	return r;"
	print "}"
}
