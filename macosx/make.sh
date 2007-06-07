#!/bin/sh
set -e

VER=$(cat ../VERSION)
PYVERS="2.4 2.5"
PMPROJ_TMPL="pytables-@VER@-py@PYVER@.pmproj"

SUBPKGS="szip-2.0.pkg hdf5-1.6.5.pkg numpy-1.0.3 SELF"

packagemaker=/Developer/Applications/Utilities/PackageMaker.app/Contents/MacOS/PackageMaker

for PYVER in $PYVERS; do
	PMPROJ=$(echo "$PMPROJ_TMPL" | sed -e "s/@VER@/$VER/" -e "s/@PYVER@/$PYVER/")
	MPKG="PyTables Pro $VER for Python $PYVER.mpkg"

	if [ "$1" = "clean" ]; then
		rm -rf "$PMPROJ" "$MPKG" *.bak
		continue
	fi

	echo "Creating $PMPROJ..."
	plutil -convert xml1 -o "$PMPROJ" "$PMPROJ_TMPL"

	sed -i .bak -e "s/@VER@/$VER/g" -e "s/@PYVER@/$PYVER/g" "$PMPROJ"

	echo "Building $MPKG..."
	# Avoiding the verbose flag makes building fail! ;(
	$packagemaker -build -proj "$PMPROJ" -p "$MPKG" -v

	echo -n "Adding subpackages..."
	for SUBPKG in $SUBPKGS; do
		echo -n " $SUBPKG"
		if [ "$SUBPKG" = "SELF" ]; then
			SUBPKG="$(echo ../dist/tables-$VER-py$PYVER*pkg)"
		elif [ $(expr "$SUBPKG" : ".*\.pkg") != 0 ]; then
			SUBPKG="../../$SUBPKG"
		else
			SUBPKG="$(echo ../../$SUBPKG/dist/$SUBPKG-py$PYVER*pkg)"
		fi
		cp -R "$SUBPKG" "$MPKG/Contents/Packages"
	done
	echo
done
echo "Done"
