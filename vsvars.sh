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

else
	echo "error: Microsoft compilers not found"
	exit 1
fi

export VSINSTALLDIR
export VCINSTALLDIR
export FrameworkDir="c:\\Windows\\Microsoft.NET\\Framework$B64"

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

# Don't care about .NET tools, libs or includes
#if [ "$WindowsSdkDir" ]
#then
#	WSD=`cygpath -u "$WindowsSdkDir"`
#	export PATH="$WSD/bin:$PATH"
#	export INCLUDE="$WindowsSdkDir\\include;$INCLUDE"
#	export LIB="$WindowsSdkDir\\lib$X64;$LIB"
#	unset WSD
#fi


#
# Root of Visual Studio ide installed files.
#
export DevEnvDir="$VSINSTALLDIR\\Common7\\IDE"

# $VCINSTALLDIR\Common7\IDE dir is added only for real setup.

DED=`cygpath -u "$DevEnvDir"`
VSD=`cygpath -u "$VSINSTALLDIR"`
VCD=`cygpath -u "$VCINSTALLDIR"`
FD=`cygpath -u "$FrameworkDir"`

export PATH="$VCD/bin/$AMD64:$VCD/VCPackages:$VSD/Common7/Tools:$VSD/Team Tools/Performance Tools$X64:$PATH"

export INCLUDE="$VCINSTALLDIR\\include;$VCINSTALLDIR\\atlmfc\\include;$INCLUDE"

export LIB="$VCINSTALLDIR\\lib\\$AMD64;$VCINSTALLDIR\\atlmfc\\lib\\$AMD64;$LIB"

export LIBPATH="$VCINSTALLDIR\\lib\\$AMD64;$VCINSTALLDIR\\atlmfc\\lib\\$AMD64;$LIBPATH"

unset DED VSD VCD FD
