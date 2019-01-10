# misc.py

from __future__ import print_function

__doc__ = """Provide object caching support."""


import cPickle as pickle
import os
import socket
import time


def ensure_dir (dirname):
    """Make sure ``dirname`` exists and is a directory."""
    if not os.path.isdir (dirname):
        try:
            os.makedirs (dirname)   # throws if exists as file
        except OSError as e:
            if e.errno != os.errno.EEXIST:
                raise
    return dirname

def save (obj, filename):
    """Dump `obj` to `filename` using the pickle module."""
    outdir, outfile = os.path.split (filename)
    save_id = '{0}_nixtime_{2:.0f}_job_{1}'.format (
        socket.gethostname (), os.getpid (), time.time ())
    temp_filename = os.path.join (outdir, '.part_{0}_id_{1}'.format (
        outfile, save_id))
    with open (temp_filename, 'wb') as f:
        pickle.dump (obj, f, -1)
    os.rename (temp_filename, filename)

def resave (obj):
    """Dump `obj` to the filename from which it was loaded."""
    save (obj, obj.__cache_source_filename)

def load (filename):
    """Load `filename` using the pickle module."""
    with open (filename) as f:
        out = pickle.load (f)
        try:
            out.__cache_source_filename = filename
        except:
            pass
        return out

def regen (filename, func, *args, **kwargs):
    """Evaluate `func` (\*`args`, \*\*`kwargs`), save to `filename`, and return
    result."""
    thing = func (*args, **kwargs)
    save (thing, filename)
    return thing

def get (filename, func, *args, **kwargs):
    """Load `filename` if it exists; otherwise generate it with func."""
    try:
        return load (filename)
    except:
        return regen (filename, func, *args, **kwargs)
