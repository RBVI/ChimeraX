BEGIN {
	state = "unknown"
}
$1 ~ /^!entry.newparm.parm.atoms$/ {
	state = "atoms"
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
#!entry.PARAMETERS.parm.atoms table  str type  dbl mass  dbl e  dbl r  int element  int hybrid  str desc
# "H" 1.0080000000 0.0200000000 1.0000000000 1 3 ""
		printf "\tAtomTypeRadii[%s] = %g;\n", $1, $4
	}
}
