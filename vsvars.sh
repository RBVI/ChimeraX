# port of VSINSTALLDIR/Common7/Tools/vsvars32.bat to bash
# and merged with Visual Studio 2008's version and 64-bit version
B64=""
AMD64=""
X64=""
if [ -e "/cygdrive/c/Program Files (x86)/Microsoft Visual Studio 12.0" ]
then
	echo Setting environment for using Microsoft Visual Studio 2013 x86 tools.
	platform=2013
	VSINSTALLDIR="c:\\Program Files (x86)\\Microsoft Visual Studio 12.0"
	if [ -d "/cygdrive/c/Program Files (x86)/Microsoft Visual Studio 12.0/VC/bin/x86_amd64" ]
	then
		B64="64"
		AMD64="x86_amd64"
		X64="\\x64"
	fi
	VCINSTALLDIR="$VSINSTALLDIR\\VC"
	FrameworkVersion=v4.0.30319
	Framework35Version=v3.5

elif [ -e "/cygdrive/c/Program Files/Microsoft Visual Studio 10.0" ]
then
	echo Setting environment for using Microsoft Visual Studio 2010 x86 tools.
	platform=2010
	if [ -d "/cygdrive/c/Program Files (x86)/Microsoft Visual Studio 10.0" ]
	then
		VSINSTALLDIR="c:\\Program Files (x86)\\Microsoft Visual Studio 10.0"
		if [ -d "/cygdrive/c/Program Files (x86)/Microsoft Visual Studio 10.0/VC/bin/amd64" ]
		then
			B64="64"
			AMD64="amd64"
			X64="\\x64"
		fi
	else
		VSINSTALLDIR="c:\\Program Files\\Microsoft Visual Studio 10.0"
	fi
	VCINSTALLDIR="$VSINSTALLDIR\\VC"
	FrameworkVersion=v4.0.30319
	Framework35Version=v3.5

elif [ -e "/cygdrive/c/Program Files/Microsoft Visual Studio 9.0" ]
then
	echo Setting environment for using Microsoft Visual Studio 2008 x86 tools.
	platform=2008
	if [ -d "/cygdrive/c/Program Files (x86)/Microsoft Visual Studio 9.0" ]
	then
		VSINSTALLDIR="c:\\Program Files (x86)\\Microsoft Visual Studio 9.0"
		if [ -d "/cygdrive/c/Program Files (x86)/Microsoft Visual Studio 9.0/VC/bin/amd64" ]
		then
			B64="64"
			AMD64="amd64"
			X64="\\x64"
		fi
	else
		VSINSTALLDIR="c:\\Program Files\\Microsoft Visual Studio 9.0"
	fi
	VCINSTALLDIR="$VSINSTALLDIR\\VC"
	FrameworkVersion=v2.0.50727
	Framework35Version=v3.5

elif [ -e "/cygdrive/c/Program Files/Microsoft Visual Studio .NET 2003" ]
then
	echo "Setting environment for using Microsoft Visual Studio .NET 2003 tools."
	platform=2003
	VSINSTALLDIR="C:\\Program Files\\Microsoft Visual Studio .NET 2003"
	VCINSTALLDIR="$VSINSTALLDIR\\Vc7"
	FrameworkVersion=v1.1.4322
else
	echo "error: Microsoft compilers not found"
	exit 1
fi

export VSINSTALLDIR
export VCINSTALLDIR
export FrameworkDir="c:\\Windows\\Microsoft.NET\\Framework$B64"

if [ $platform == 2003 ]
then
	FrameworkSDKDir="$VSINSTALLDIR\\SDK\\v1.1"
	FSD=`cygpath -u "$FrameworkSDKDir"`
	export PATH="$FSD/bin:$PATH"
	export INCLUDE="$FrameworkSDKDir\\include;$INCLUDE"
	export LIB="$FrameworkSDKDir\\lib;$LIB"
	unset FSD
fi

function GetWindowsSdkDir () {
	GetWindowsSdkDirHelper HKLM > /dev/null 2>&1
	if [ $? -ne 0 ]
	then
		GetWindowsSdkDirHelper HKCU > /dev/null 2>&1
		if [ $? -ne 0 ]
		then
			# VS2003 location
			export WindowsSdkDir="$VCINSTALLDIR\\PlatformSDK"
		fi
	fi
	return 0
}

function GetWindowsSdkDirHelper () {
	i=`regtool get "/$1/SOFTWARE/Microsoft/Microsoft SDKs/Windows/CurrentInstallFolder"`
	if [ "$i" ]
	then
		export WindowsSdkDir="$i"
		return 0
	fi
	return 1
}

GetWindowsSdkDir

if [ "$WindowsSdkDir" ]
then
	WSD=`cygpath -u "$WindowsSdkDir"`
	export PATH="$WSD/bin:$PATH"
	export INCLUDE="$WindowsSdkDir\\include;$INCLUDE"
	export LIB="$WindowsSdkDir\\lib$X64;$LIB"
	unset WSD
fi


#
# Root of Visual Studio ide installed files.
#
export DevEnvDir="$VSINSTALLDIR\\Common7\\IDE"

# $VCINSTALLDIR\Common7\Tools dir is added only for real setup.

DED=`cygpath -u "$DevEnvDir"`
VSD=`cygpath -u "$VSINSTALLDIR"`
VCD=`cygpath -u "$VCINSTALLDIR"`
FD=`cygpath -u "$FrameworkDir"`

export PATH="$VCD/VCPackages:$DED:$VCD/bin:$VSD/Common7/Tools:$VSD/Common7/Tools/bin:$FD/v3.5:$FD/$FrameworkVersion:$PATH"
export INCLUDE="$VCINSTALLDIR\\atlmfc\\include;$VCINSTALLDIR\\include;$INCLUDE"
export LIB="$VCINSTALLDIR\\atlmfc\\lib\\$AMD64;$VCINSTALLDIR\\lib\\$AMD64;$LIB"
export LIBPATH="$FrameworkDir\\v3.5;$FrameworkDir\\$FrameworkVersion;$VCINSTALLDIR\\atlmfc\\lib\\$AMD64;$VCINSTALLDIR\\lib\\$AMD64;$LIBPATH"
if [ "$B64" ]
then
	export PATH="$VCD/bin/$AMD64:$PATH"
fi

unset DED VSD VCD FD
