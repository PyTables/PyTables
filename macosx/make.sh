#!/bin/sh
set -e
set -x

PMPROJ_TMPL="pytables-@VER@-py@PYVER@.pmproj"
WELCOME_TMPL="welcome-@VER@-py@PYVER@.rtf"
BACKGROUND="background.tif"
SUBPKGS="hdf5-1.6.7.pkg numpy-1.0.4"

VER=$(cat ../VERSION)
WELCOME_EXT=$(echo "$WELCOME_TMPL" | sed -ne 's/.*\.\(.*\)/\1/p')
SUBPKGS="SELF $SUBPKGS"

if [ "$1" = "clean" ]; then
	cleaning=true
	PYVERS="$(ls /Library/Frameworks/Python.framework/Versions | grep -v Current)"
else
	PYVERS="$(ls -d ../dist/tables-$VER-py*-*.?pkg | sed -e 's#.*-py\([0-9].[0-9]\)-.*#\1#')"
	if [ ! "$PYVERS" ]; then
		echo "No available binary packages." > /dev/stderr
		exit 1
	fi
fi


packagemaker=/Developer/Applications/Utilities/PackageMaker.app/Contents/MacOS/PackageMaker

for PYVER in $PYVERS; do
        PMPROJ=$(echo "$PMPROJ_TMPL" | sed -e "s/@VER@/$VER/" -e "s/@PYVER@/$PYVER/")
        WELCOME=$(echo "$WELCOME_TMPL" | sed -e "s/@VER@/$VER/" -e "s/@PYVER@/$PYVER/")
        MPKG="PyTables $VER for Python $PYVER.mpkg"
        LICENSE="$MPKG/Contents/Resources/License.txt"
        DMGDIR="PyTables $VER (py$PYVER)"
        DMG="PyTables-${VER}.macosxppc-py${PYVER}.dmg"

        if [ $cleaning ]; then
                rm -rf "$WELCOME" "$PMPROJ" "$MPKG" "$DMGDIR" "$DMG" *.bak
                continue
        fi

        echo "Creating $WELCOME..."
        sed -e "s/@VER@/$VER/g" -e "s/@PYVER@/$PYVER/g" < "$WELCOME_TMPL" > "$WELCOME"

        echo "Creating $PMPROJ..."
        plutil -convert xml1 -o "$PMPROJ" "$PMPROJ_TMPL"
        sed -e "s/@VER@/$VER/g" -e "s/@PYVER@/$PYVER/g" "$PMPROJ"

        echo "Building $MPKG..."
        # Avoiding the verbose flag makes building fail! ;(
        $packagemaker -i pytables -build -proj "$PMPROJ" -p "$MPKG" -v

        echo "Fixing $MPKG..."
        cp "$WELCOME" "$MPKG/Contents/Resources/Welcome.$WELCOME_EXT"
        cp "$BACKGROUND" "$MPKG/Contents/Resources"

        echo -n "Adding subpackages..."
        true > "$LICENSE"
        for SUBPKG in $SUBPKGS; do
                echo -n " $SUBPKG"
                if [ "$SUBPKG" = "SELF" ]; then
                        PKGSRC="$(echo ../dist/tables-$VER-py$PYVER*pkg)"
                elif [ $(expr "$SUBPKG" : ".*\.pkg") != 0 ]; then
                        PKGSRC="../../$SUBPKG"
                else
                        PKGSRC="$(echo ../../$SUBPKG/dist/$SUBPKG-py$PYVER*pkg)"
                fi
                cp -R "$PKGSRC" "$MPKG/Contents/Packages"

                PKGRES="$PKGSRC/Contents/Resources"
                if [ "$SUBPKG" = "SELF" ]; then
                        # Place the proper license in all PyTables packages.
                        PKGDST="$MPKG/Contents/Packages/$(basename $PKGSRC)"
                        find "$PKGDST" -path "*/Contents/Resources" -type d \
                                -exec cp "../LICENSE-$LIC.txt" '{}/License.txt' ';'
                        cat "../LICENSE-$LIC.txt" >> "$LICENSE"
                elif test -f "$PKGRES/License.txt"; then
                        cat $_ >> "$LICENSE"
                else
                        cat "$PKGRES/English.lproj/License.txt" >> "$LICENSE"
                fi
                echo -e "\n--------------------------------\n" >> "$LICENSE"
        done
        echo

        echo "Building $DMG..."
        mkdir -p "$DMGDIR"
        cp "$LICENSE" "$DMGDIR/License.txt"
        mv "$MPKG" "$DMGDIR"
        cp ../README.txt "$DMGDIR/ReadMe.txt"
        mkdir -p "$DMGDIR/Examples"
        cp -R ../examples/* "$DMGDIR/Examples/"
        cp ../doc/usersguide.pdf "$DMGDIR/User's Guide.pdf"
        cp -R ../doc/html "$DMGDIR/User's Guide (HTML)"
        hdiutil create -srcfolder "$DMGDIR" -anyowners -format UDZO -imagekey zlib-level=9 "$DMG"
done
echo "Done"
