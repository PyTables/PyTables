#!/bin/sh

# Script to automatically subtree merge a specifc version of blosc to
# python-blosc.

# TODO
# ----
#
# * Should probably check working tree and index are clean.

# configure remote
remote="git://github.com/FrancescAlted/blosc.git"

# check argument
if [ -z "$1" ] ; then
    echo "usage: subtree-merge-blosc.sh <blosc-tag>"
    exit 1
fi

# extract the blosc tag the user has requested
blosc_tag="$1"
blosc_tag_long="refs/tags/$1"

# check that it exists on the remote side
remote_ans=$( git ls-remote $remote $blosc_tag_long )
if [ -z "$remote_ans" ] ; then
    echo "no remote tag '$1' found"
    exit 1
else
    echo "found remote tag: '$remote_ans'"
fi

# fetch the contents of this tag
git fetch $remote $blosc_tag_long || exit 1
# subtree merge it
git merge --squash -s subtree FETCH_HEAD || exit 1
if git diff --staged --quiet ; then
    echo "nothing new to be committed"
    exit 1
else
    # set a custom commit message
    git commit -m "subtree merge blosc $blosc_tag" || exit 1
fi
