#!/bin/sh
set -e

PYVERS="2.4 2.5"
PMPROJ_TMPL="pytables-@VER@-py@PYVER@.pmproj"
WELCOME_TMPL="welcome-@VER@-py@PYVER@.rtf"
BACKGROUND="background.tif"
SUBPKGS="hdf5-1.6.5.pkg numpy-1.0.3"

VER=$(cat ../VERSION)
VERNP=${VER%pro}
WELCOME_EXT=$(echo "$WELCOME_TMPL" | sed -ne 's/.*\.\(.*\)/\1/p')
SUBPKGS="SELF $SUBPKGS"

packagemaker=/Developer/Applications/Utilities/PackageMaker.app/Contents/MacOS/PackageMaker

for PYVER in $PYVERS; do
	PMPROJ=$(echo "$PMPROJ_TMPL" | sed -e "s/@VER@/$VER/" -e "s/@VERNP@/$VERNP/" -e "s/@PYVER@/$PYVER/")
	WELCOME=$(echo "$WELCOME_TMPL" | sed -e "s/@VER@/$VER/" -e "s/@VERNP@/$VERNP/" -e "s/@PYVER@/$PYVER/")
	MPKG="PyTables Pro $VERNP for Python $PYVER.mpkg"

	if [ "$1" = "clean" ]; then
		rm -rf "$WELCOME" "$PMPROJ" "$MPKG" *.bak
		continue
	fi

	echo "Creating $WELCOME..."
	sed -e "s/@VER@/$VER/g" -e "s/@VERNP@/$VERNP/g" -e "s/@PYVER@/$PYVER/g" < "$WELCOME_TMPL" > "$WELCOME"

	echo "Creating $PMPROJ..."
	plutil -convert xml1 -o "$PMPROJ" "$PMPROJ_TMPL"
	sed -i .bak -e "s/@VER@/$VER/g" -e "s/@VERNP@/$VERNP/g" -e "s/@PYVER@/$PYVER/g" "$PMPROJ"

	echo "Building $MPKG..."
	# Avoiding the verbose flag makes building fail! ;(
	$packagemaker -build -proj "$PMPROJ" -p "$MPKG" -v

	echo "Fixing $MPKG..."
	cp "$WELCOME" "$MPKG/Contents/Resources/Welcome.$WELCOME_EXT"
	cp "$BACKGROUND" "$MPKG/Contents/Resources"

	echo -n "Adding subpackages..."
	LICENSE="$MPKG/Contents/Resources/License.txt"
	true > "$LICENSE"
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

		PKGRES="$SUBPKG/Contents/Resources"
		if test -f "$PKGRES/License.txt"; then
			cat $_ >> "$LICENSE"
		else
			cat "$PKGRES/English.lproj/License.txt" >> "$LICENSE"
		fi
		echo -e "\n--------------------------------\n" >> "$LICENSE"
	done
	echo
done
echo "Done"
