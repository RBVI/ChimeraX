# vim: syntax=bash:
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
if [ ! -z "$AUTOMATIC_DISCOVERY" ]
then
	if [ -e "$(cygpath --absolute C:/Program\ Files\ \(x86\)/Microsoft\ Visual\ Studio/Installer)" ]
	then 
		export VSWHEREPATH="$(cygpath --absolute C:/Program\ Files\ \(x86\)/Microsoft\ Visual\ Studio/Installer)"
		if ! command -v vswhere &> /dev/null
		then
			export PATH="$PATH:$VSWHEREPATH"
		fi
		
		VCTBinDirCRLF="$(vswhere.exe -find "VC/Tools/MSVC/*/bin/Hostx64/x64")"
		VCTRedistDirCRLF="$(vswhere.exe -find "VC/Redist/MSVC/*/x64")"
		VCTLibDirCRLF="$(vswhere.exe -find "VC/Tools/MSVC/*/lib/x64")"
		VCTATLMFCLibDirCRLF="$(vswhere.exe -find "VC/Tools/MSVC/*/atlmfc/lib/x64")"
		VCTATLMFCIncDirCRLF="$(vswhere.exe -find "VC/Tools/MSVC/*/atlmfc/include")"
		VCTIncludeDirCRLF="$(vswhere.exe -find "VC/Tools/MSVC/*/include")"
		VCTRefDirCRLF="$(vswhere.exe -find "VC/Tools/MSVC/*/lib/x86/store/references")"
		CRTPathCRLF="$(vswhere.exe -find "VC/Redist/MSVC/*/x64/Microsoft*CRT")"
		WindowsSDKDirCRLF="$(regtool -W get '/HKLM/SOFTWARE/Microsoft/Microsoft SDKs/Windows/v10.0/InstallationFolder')"
		WindowsSDKVerCRLF="$(regtool -W get '/HKLM/SOFTWARE/Microsoft/Microsoft SDKs/Windows/v10.0/ProductVersion')"
		WindowsSDKVer="${WindowsSDKVerCRLF%$'\r'}.0"
		
		export VCTInstallDir="$(cygpath --absolute "${VCTBinDirCRLF%$'\r'}")"
		export VCTRedistDir="$(cygpath -u "${VCTRedistDirCRLF%$'\r'}")"
		export VCTLibDir="$(cygpath -w "${VCTLibDirCRLF%$'\r'}")"
		export VCTATLMFCLibDir="$(cygpath -w "${VCTATLMFCLibDirCRLF%$'\r'}")"
		export VCTATLMFCIncDir="$(cygpath -w "${VCTATLMFCIncDirCRLF%$'\r'}")"
		export VCTIncludeDir="$(cygpath -w "${VCTIncludeDirCRLF%$'\r'}")"
		export VCTRefDir="$(cygpath -w "${VCTRefDirCRLF%$'\r'}")"
		
		export WindowsSDKUmDir="$(cygpath -w "${WindowsSDKDirCRLF}Lib\\${WindowsSDKVer}\um\x64")"
		export WindowsSDKUmIncDir="$(cygpath -w "${WindowsSDKDirCRLF}Include\\${WindowsSDKVer}\um")"
		export WindowsSDKUCRTDir="$(cygpath -w "${WindowsSDKDirCRLF}Lib\\${WindowsSDKVer}\ucrt\x64")"
		export WindowsSDKUCRTIncDir="$(cygpath -w "${WindowsSDKDirCRLF}Include\\${WindowsSDKVer}\ucrt")"
		export WindowsSDKSharedIncDir="$(cygpath -w "${WindowsSDKDirCRLF}Include\\${WindowsSDKVer}\shared")"
		export WindowsSDKWinRTIncDir="$(cygpath -w "${WindowsSDKDirCRLF}Include\\${WindowsSDKVer}\winrt")"
		export WindowsSDKCPPWinRTIncDir="$(cygpath -w "${WindowsSDKDirCRLF}Include\\${WindowsSDKVer}\cppwinrt")"
		
		export WindowsSDKLibRefDir="$(cygpath -w "${WindowsSDKDirCRLF}References\\${WindowsSDKVer}")"
		export WindowsSDKLibUMDir="$(cygpath -w "${WindowsSDKDirCRLF}UnionMetadata${WindowsSDKVer}")"

		export WindowsSDKBinDir="$(cygpath -u "${WindowsSDKDirCRLF}bin\\${WindowsSDKVer}\x64")"
		export CRTPath="$(cygpath -u "${CRTPathCRLF%$'\r'}")"
		
		export NEW_STYLE_VSVARS=1
		export VSINSTALLDIR=1

		export PATH="${VCTInstallDir}:${WindowsSDKBinDir}:$PATH"

		export INCLUDE="${VCTIncludeDir};${VCTATLMFCIncDir};${WindowsSDKUmIncDir};${WindowsSDKUCRTIncDir};${WindowsSDKSharedIncDir};${WindowsSDKWinRTIncDir};${WindowsSDKCPPWinRTIncDir};$INCLUDE"
		export LIB="${VCTLibDir};${VCTATLMFCLibDir};${WindowsSDKUCRTDir};${WindowsSDKUmDir};$LIB"
		export LIBPATH="${VCTLibDir};${VCTATLMFCLibDir};${VCTRefDir};${WindowsSDKLibRefDir};${WindowsSDKLibUMDir};$LIBPATH"

		return 0
	else
		echo "error: Could not find vswhere.exe to automatically set vsvars"
		return 1
	fi
else
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
		return 1
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
fi
