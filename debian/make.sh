VER=$(cat ../VERSION)
VERNP=${VER%pro}
LICENSES="$(ls ../LICENSE-*.txt | sed -e 's#.*-\([a-z]*\).txt$#\1#')"
DEBFILES="control changelog rules"

echo $LICENSES

for license in $LICENSES ; do
    cp ../LICENSE-$license.txt ../LICENSE.txt
    for f in $DEBFILES ; do
        cat $f.in | sed -e "s/@LICENSE@/$license/g" > $f
            if [ $f == "rules" ] ; then
                chmod 755 $f
	    fi
    done
    pkgname=pytables-pro-$license-$VERNP
    ( cd .. ; python setup.py sdist )
    ( cd ../dist ; tar xvfz tables-$VER.tar.gz ; \
      mv tables-$VER $pkgname ; \
      tar cvfz $pkgname.tar.gz $pkgname ; \
      rm -rf $pkgname ; rm tables-$VER.tar.gz )
    rm ../LICENSE.txt
done
