#!/bin/sh

# Script to automatically subtree merge a specific version of blosc.
#
# For more information, see:
# http://github.com/Blosc/subtree-merge-blosc
#
# Please do not change this file and submit any changes to the above URL
# instead.
#
# Author: Valentin Haenel <valentin@haenel.co>
#
# License: WTFPL (do what the fuck you want public license)

version=0.12

# configure remote
remote="git://github.com/Blosc/c-blosc.git"
# regular expression for tag
tag_regex="^v[0-9]*\.[0-9]*\.[0-9]*$"

fatal () {
    echo $1
    exit 1
}

# check argument
if [ -z "$1" ] ; then
    fatal "usage: subtree-merge-blosc.sh <blosc-tag>"
fi

# check we are in the repository root dir
if [ "$( git rev-parse --show-toplevel)" != "$( pwd )" ] ; then
    fatal "not currently in the repositories top-level"
fi

# check if working directory clean
status="$( git status --porcelain --ignored )"
if [ "$status" != "" ] ; then
    git status --ignored
    fatal "working directory is not clean, see above for details"
fi

# check c-blosc subdirectory exists
if ! [ -d "c-blosc" ] ; then
    fatal "'c-blosc' subdirectory doesn't exist"
fi

# extract the blosc tag the user has requested
blosc_tag="$1"

# check that the tag is sane
if ! echo $blosc_tag | grep -q $tag_regex ; then
    fatal "Tag: '$1' doesn't match regex '$tag_regex'"
fi
blosc_tag_long="refs/tags/$1"

# check that it exists on the remote side
remote_ans=$( git ls-remote $remote $blosc_tag_long )
if [ -z "$remote_ans" ] ; then
    fatal "no remote tag '$1' found"
else
    echo "found remote tag: '$remote_ans'"
fi

# fetch the contents of this tag
git fetch $remote $blosc_tag_long || exit 1
# subtree merge it
#git merge --squash -s subtree FETCH_HEAD || exit 1
# if this doesn't work you could use:
 git merge --squash -Xsubtree=c-blosc -Xtheirs FETCH_HEAD || exit 1
if git diff --staged --quiet ; then
    fatal "nothing new to be committed"
else
    # set a custom commit message
    git commit -m "subtree merge blosc $blosc_tag" || exit 1
fi
