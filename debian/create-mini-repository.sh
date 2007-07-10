rm -rf pytables-pro-*-2*
mkdir binary source
mv *.deb binary
mv pytables* source
dpkg-scanpackages binary /dev/null | gzip -9c > Packages.gz
dpkg-scansources source /dev/null | gzip -9c > Sources.gz
