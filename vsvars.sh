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
X64=""
if [ -e "/cygdrive/c/Program Files (x86)/Microsoft Visual Studio/2019" ]
then
	echo Setting environment for using Microsoft Visual Studio 2019
	Platform=x64
	VSINSTALLDIR="c:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community"
	WindowsSdkDir="c:\\Program Files (x86)\\Windows Kits\\10"
	ExtensionSdkDir="$WindowsSdkDir\\ExtensionSDKs"
	# also use 64 bit tools
	B64=64
	X64="\\x64"

	Framework40Version=v4.0
	FrameworkVersion=v4.0.30319
	FrameworkVersion64=v4.0.30319
	VCToolsVersion=14.24.28314
	VisualStudioVersion=16.0
	WindowsSDKLibVersion=10.0.18362.0
	WindowsSDKVersion=10.0.18362.0

	VCINSTALLDIR="$VSINSTALLDIR\\VC"
	VCToolsInstallDir="$VCINSTALLDIR\\Tools\\MSVC\\$VCToolsVersion"
	VCToolsRedistDir="$VCINSTALLDIR\\Redist\MSVC\\14.24.28127"
elif [ -e "/cygdrive/c/Program Files (x86)/Microsoft Visual Studio/2017" ]
then
	echo Setting environment for using Microsoft Visual Studio 2017
	Platform=x64
	VSINSTALLDIR="c:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community"
	WindowsSdkDir="c:\\Program Files (x86)\\Windows Kits\\10"
	ExtensionSdkDir="$WindowsSdkDir\\ExtensionSDKs"
	# also use 64 bit tools
	B64=64
	X64="\\x64"

	Framework40Version=v4.0
	FrameworkVersion=v4.0.30319
	FrameworkVersion64=v4.0.30319
	VCToolsVersion=14.16.27023
	VisualStudioVersion=15.0
	WindowsSDKLibVersion=10.0.17763.0
	WindowsSDKVersion=10.0.17763.0

	VCINSTALLDIR="$VSINSTALLDIR\\VC"
	VCToolsInstallDir="$VCINSTALLDIR\\Tools\\MSVC\\$VCToolsVersion"
	VCToolsRedistDir="$VCINSTALLDIR\\Redist\MSVC\\14.16.27012"
else
	echo "error: neither Microsoft Visual C++ 2019 nor 2017 compiler found"
	exit 1
fi

export VSINSTALLDIR
export VCINSTALLDIR
export VCToolsInstallDir
export VCToolsRedistDir
export FrameworkDir="c:\\Windows\\Microsoft.NET\\Framework$B64"
export VisualStudioVersion

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
VCT=`cygpath -u "$VCToolsInstallDir"`
FD=`cygpath -u "$FrameworkDir"`
WSD=`cygpath -u "$WindowsSdkDir"`

export PATH="$VCT/bin/HostX64/x64:\
$WSD/bin/x64:\
$VSD/Team Tools/Performance Tools/x64:\
$VSD/Team Tools/Performance Tools:\
$WSD/bin/$WindowsSDKVersion/x64:\
$WSD/bin/x64:\
$VSD/Common7/Tools:\
$PATH"

export INCLUDE="$VCToolsInstallDir\\include;\
$VCToolsInstallDir\\atlmfc\\include;\
$WindowsSdkDir\\include\\$WindowsSDKVersion\\ucrt;\
$WindowsSdkDir\\include\\$WindowsSDKVersion\\shared;\
$WindowsSdkDir\\include\\$WindowsSDKVersion\\um;\
$WindowsSdkDir\\include\\$WindowsSDKVersion\\winrt;\
$WindowsSdkDir\\include\\$WindowsSDKVersion\\cppwinrt;\
$INCLUDE"

export LIB="$VCToolsInstallDir\\lib\\$X64;\
$VCToolsInstallDir\\atlmfc\\lib\\$X64;\
$WindowsSdkDir\\lib\\$WindowsSDKLibVersion\\ucrt$X64;\
$WindowsSdkDir\\lib\\$WindowsSDKLibVersion\\um$X64;\
$LIB"

export LIBPATH="$VCToolsInstallDir\\lib\\$X64;\
$VCToolsInstallDir\\atlmfc\\lib\\$X64;\
$VCToolsInstallDir\\lib\\x86\\store\\references\\$X64;\
$WindowsSdkDir\\UnionMetadata\\$WindowsSDKVersion;\
$WindowsSdkDir\\References\\$WindowsSDKVersion;\
$LIBPATH"

unset DED VSD VCD FD WSD VCT
