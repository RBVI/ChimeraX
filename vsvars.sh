# === UCSF ChimeraX Copyright ===
# Copyright 2016 Regents of the University of California.
# All rights reserved.  This software provided pursuant to a
# license agreement containing restrictions on its disclosure,
# duplication and use.  For details see:
# http://www.rbvi.ucsf.edu/chimerax/docs/licensing.html
# This notice must be embedded in or attached to all copies,
# including partial copies, of the software or any revisions
# or derivations thereof.
# === UCSF ChimeraX Copyright ===

# port of VSINSTALLDIR/Common7/Tools/vsvars32.bat to bash
# and merged with Visual Studio 2008's version and 64-bit version
B64=""
AMD64=""
X64=""
if [ -e "/cygdrive/c/Program Files (x86)/Microsoft Visual Studio 14.0" ]
then
	echo Setting environment for using Microsoft Visual Studio 2015
	platform=2015
	VSINSTALLDIR="c:\\Program Files (x86)\\Microsoft Visual Studio 14.0"
	WindowsSdkDir="c:\\Program Files (x86)\\Windows Kits\\10"
	MicrosoftSdkDir="c:\Program Files (x86)\Microsoft SDKs\Windows Kits\10"
	if [ -d "/cygdrive/c/Program Files (x86)/Microsoft Visual Studio 14.0/VC/bin/amd64" ]
	then
		echo Using native 64 bit tools
		B64="64"
		AMD64="amd64"
		X64="\\x64"
	else
		echo Using native 32 bit tools
	fi
	VCINSTALLDIR="$VSINSTALLDIR\\VC"
	FrameworkVersion=v4.0.30319
	FrameworkVersion64=v4.0.30319
	Framework40Version=v4.0
	VisualStudioVersion=14.0
	WindowsSDKLibVersion=10.0.10586.0
	WindowsSDKVersion=10.0.10586.0

else
	echo "error: Microsoft compilers not found"
	exit 1
fi

export VSINSTALLDIR
export VCINSTALLDIR
export FrameworkDir="c:\\Windows\\Microsoft.NET\\Framework$B64"

function GetMicrosoftSdkDir () {
	GetMicrosoftSdkDirHelper HKLM > /dev/null 2>&1
	if [ $? -ne 0 ]
	then
		GetMicrosoftSdkDirHelper HKCU > /dev/null 2>&1
		if [ $? -ne 0 ]
		then
			# VS2003 location
			export MicrosoftSdkDir="$VCINSTALLDIR\\PlatformSDK"
		fi
	fi
	return 0
}

function GetMicrosoftSdkDirHelper () {
	i=`regtool get "/$1/SOFTWARE/Microsoft/Microsoft SDKs/Windows/CurrentInstallFolder"`
	if [ "$i" ]
	then
		export MicrosoftSdkDir="$i"
		return 0
	fi
	return 1
}

GetMicrosoftSdkDir

# Don't care about .NET tools, libs or includes
#if [ "$MicrosoftSdkDir" ]
#then
#	WSD=`cygpath -u "$MicrosoftSdkDir"`
#	export PATH="$WSD/bin:$PATH"
#	export INCLUDE="$MicrosoftSdkDir\\include;$INCLUDE"
#	export LIB="$MicrosoftSdkDir\\lib$X64;$LIB"
#	unset WSD
#fi


#
# Root of Visual Studio ide installed files.
#
export DevEnvDir="$VSINSTALLDIR\\Common7\\IDE"

DED=`cygpath -u "$DevEnvDir"`
VSD=`cygpath -u "$VSINSTALLDIR"`
VCD=`cygpath -u "$VCINSTALLDIR"`
FD=`cygpath -u "$FrameworkDir"`
WSD=`cygpath -u "$WindowsSdkDir"`

export PATH="$VCD/bin/$AMD64:\
$WSD/bin/x64:\
$VCD/VCPackages:\
$VSD/Common7/IDE:\
$VSD/Common7/Tools:\
$VSD/Team Tools/Performance Tools/x64:\
$PATH"

export INCLUDE="$VCINSTALLDIR\\include;\
$VCINSTALLDIR\\atlmfc\\include;\
$WindowsSdkDir\\include\\$WindowsSDKVersion\\ucrt;\
$WindowsSdkDir\\include\\$WindowsSDKVersion\\shared;\
$WindowsSdkDir\\include\\$WindowsSDKVersion\\um;\
$WindowsSdkDir\\include\\$WindowsSDKVersion\\winrt;\
$INCLUDE"
#$UniversalCRTSdkDir\\include\\$UCRTVersion\\ucrt

export LIB="$VCINSTALLDIR\\lib\\$AMD64;\
$VCINSTALLDIR\\atlmfc\\lib\\$AMD64;\
$WindowsSdkDir\\lib\\$WindowsSDKLibVersion\\ucrt$X64;\
$WindowsSdkDir\\lib\\$WindowsSDKLibVersion\\um$X64;\
$LIB"

export LIBPATH="$VCINSTALLDIR\\lib\\$AMD64;\
$VCINSTALLDIR\\atlmfc\\lib\\$AMD64;\
$WindowsSdkDir\\UnionMetadata;\
$WindowsSdkDir\\References;\
$MicrosoftSdkDir\\ExtensionSDKs\\Microsoft.VCLibs\\$VisualStudioVersion\\References\\CommonConfiguration\neutral;\
$LIBPATH"

unset DED VSD VCD FD WSD
