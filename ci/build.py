import os
import re
import sys
import subprocess
import platform

heavy = os.environ.get("HEAVY", "false")
venv = os.environ.get("VIRTUALENV", "default")

if platform.system() == "Windows":
    home = "C:\\jenkins"
    bin = "Scripts"
    exe = ".exe"
else:
    home = os.path.expanduser("~")
    bin = "bin"
    exe = ""

bindir = os.path.join(home, venv, bin)
python = os.path.join(bindir, "python" + exe)
pip = os.path.join(bindir, "pip" + exe)
nosetests = os.path.join(bindir, "nosetests" + exe)
coverage = os.path.join(bindir, "coverage" + exe )

print "Using %s..." % python

os.environ["PATH"] = os.path.pathsep.join([bindir, os.environ["PATH"]])
try:
  os.environ["PYTHONPATH"] = os.path.pathsep.join([".", os.environ["PYTHONPATH"]])
except KeyError:
  os.environ["PYTHONPATH"] = "."

def call(*args):
  cmd = list(args)
  rc = subprocess.call(cmd)
  if rc != 0:
    print >>sys.stderr, "FAILED: '%s'" % " ".join(cmd)
    sys.exit(rc)

call(pip, 'install', 'unittest2')

if "HDF5_DIR" in os.environ:
  call(python, 'setup.py', 'build_ext', '--inplace')
else:
  call(python, 'setup.py', 'build_ext', '--inplace', '--hdf5=/usr')

TESTS = 'tables/tests/test_all.py'

# Following led to Out of Memory for the whole Windows VM!
# call(nosetests, '--with-xunit', '--with-coverage', TESTS)

if heavy.lower() == "true":
    call(coverage, 'run', TESTS, '--verbose', '--heavy')
else:
    call(coverage, 'run', TESTS, '--verbose')
call(coverage, 'xml')
