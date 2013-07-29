#!/bin/csh -fb
set usage = "usage: $0 (amino|namino|camino|nucleic|general|ions)"
if ($#argv != 1) then
	echo "$usage"
	exit 1
endif
switch ($1)
case amino:
	set file = all_amino94.lib
	set residues = (ALA ARG ASH ASN ASP CYM CYS CYX GLH GLN GLU GLY HID HIE HIP ILE LEU LYN LYS MET PHE PRO SER THR TRP TYR VAL)
	set identifiers = ( $residues[*] )
	set outres = (ALA ARG ASH ASN ASP CYM CYS CYX GLH GLN GLU GLY HID HIE HIP ILE LEU LYN LYS MET PHE PRO SER THR TRP TYR VAL)
	set location = middle
	breaksw
case namino:
	set file = all_aminont94.lib
	set residues = (ACE NALA NARG NASN NASP NCYS NCYX NGLN NGLU NGLY NHID NHIE NHIP NILE NLEU NLYS NMET NPHE NPRO NSER NTHR NTRP NTYR NVAL)
	set identifiers = ( $residues[*] )
	set outres = (ACE ALA ARG ASN ASP CYS CYX GLN GLU GLY HID HIE HIP ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL)
	set location = start
	breaksw
case camino:
	set file = all_aminoct94.lib
	set residues = (CALA CARG CASN CASP CCYS CCYX CGLN CGLU CGLY CHID CHIE CHIP CILE CLEU CLYS CMET CPHE CPRO CSER CTHR CTRP CTYR CVAL NME)
	set identifiers = ( $residues[*] )
	set outres = (ALA ARG ASN ASP CYS CYX GLN GLU GLY HID HIE HIP ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL NME)
	set location = end
	breaksw
case nucleic:
	set file = all_nucleic94.lib
	#set residues = (DA DA3 DA5 DAN DC DC3 DC5 DCN DG DG3 DG5 DGN DT DT3 DT5 DTN RA RA3 RA5 RAN RC RC3 RC5 RCN RG RG3 RG5 RGN RU RU3 RU5 RUN)
	#set residues = (DA3 DC3 DG3 DT3 RU3 DA DC DG DT RU DA5 DC5 DG5 DT5 RU5)
	#set outres = (A C G T U A C G T U A C G T U)
	#set location = (start start start start start middle middle middle middle middle end end end end end)
	set residues = (DA3 DC3 DG3 DT3 RA3 RC3 RG3 RU3 DA DC DG DT RA RC RG RU DA5 DC5 DG5 DT5 RA5 RC5 RG5 RU5)
	set identifiers = ( $residues[*] )
	set outres = (DA DC DG DT A C G U DA DC DG DT A C G U DA DC DG DT A C G U)
	set location = (start start start start start start start start middle middle middle middle middle middle middle middle end end end end end end end end)
	breaksw
case general:
	set file = water.lib
	set residues = (TIP3)
	set identifiers = ( $residues[*] )
	set outres = (WAT)
	set location = middle
	breaksw
case ions:
	set file = ions94.lib
	set residues = (Cs\\\\+ K\\\\+ Li\\\\+ Na\\\\+ Rb\\\\+)
	set identifiers = (Cs K Li Na Rb)
	set outres = (Cs+ K+ Li+ Na+ Rb+)
	set location = middle
	breaksw
default:
	echo "$usage"
	exit 1
endsw
if ($#residues != $#outres) then
	echo "internal error1:" $#residues "!=" $#outres
	exit 1
endif
if ($#residues != $#identifiers) then
	echo "internal error2:" $#residues "!=" $#identifiers
	exit 1
endif
if ($#location != 1 && $#location != $#residues) then
	echo "internal error3:" $#location "!=" $#residues
endif
if ($#location == 1) then
	set tmp = ()
	@ i = 1
	while ($i <= $#residues)
		set tmp = ($tmp $location)
		@ i++
	end
	set location = ($tmp)
endif
echo '#include "resinternal.h"' > $1.cpp
echo '' >> $1.cpp
@ i = 1
while ($i <= $#residues)
	sed -e s/XXX/$residues[$i]/g -e s/YYY/$outres[$i]/g -e s/ZZZ/$identifiers[$i]/g < extract.awk > $$.awk
	gzcat $file.gz | awk -f $$.awk >> $1.cpp
	@ i++
end
cat << EOF >> $1.cpp

void
restmpl_init_$1(ResInitMap *rim)
{
EOF
@ i = 1
while ($i <= $#residues)
	set o = $outres[$i]
	set l = $location[$i]
	set r = $residues[$i]
	echo "	(*rim)[std::string("\""$o"\"")].$l = init_$identifiers[$i];" >> $1.cpp
	@ i++
end
echo '}' >> $1.cpp
rm $$.awk
