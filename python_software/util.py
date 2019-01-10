# util.py for grbllh

import logging
import datetime
import numpy as np


from _grbllh import Interp1D, Interp2D

def _Interp1D_eval_array (self, array):
    e = np.vectorize (lambda x: self.eval (x))
    return e (array);

def _Interp2D_eval_array (self, a, b):
    e = np.vectorize (lambda x, y: self.eval (x, y))
    return e (a, b);

Interp1D.__call__ = _Interp1D_eval_array
Interp2D.__call__ = _Interp2D_eval_array

def _get (o, key, cast=None):
    """Get member key from object o.

    First subscript (o[key]) and then member (getattr (o, key)) access will
    be tried.

    """
    try:
        if cast is not None:
            return cast (o[key])
        else:
            return o[key]
    except:
        try:
            if cast is not None:
                return cast (getattr (o, key))
            else:
                return getattr (o, key)
        except:
            logging.warning ('could not find "{0}" key in {1}'.format (key, o))
            return None

def _prep (a, dtype=float):
    return map (dtype, np.asarray (a, dtype))

def _get_list (thing):
    if isinstance (thing, list):
        return thing
    else:
        thing = [thing]
        return thing

@np.vectorize
def _timedelta_in_seconds (dt):
    """Get a timedelta in seconds."""
    return 1e-6 * (
            ((dt.days * 86400 * 1e6) + dt.seconds * 1e6) + dt.microseconds)

def _get_random_directions (N, upgoing):
    """Get N random directions.  If `upgoing`, zenith > pi/2."""
    mc_zenith_4pi = np.arccos (2 * np.random.random (N) - 1)
    if upgoing:
        zenith = np.abs (pi/2 - mc_zenith_4pi) + pi/2
    else:
        zenith = mc_zenith_4pi
    azimuth = 2 * pi * np.random.random (N)
    return zenith, azimuth

def opening_angle (zenith1, azimuth1, zenith2, azimuth2):
    """Calculate the opening angle between directions 1 and 2."""
    from numpy import sin, cos, arccos
    sin_z1 = sin (zenith1)
    cos_z1 = cos (zenith1)
    sin_z2 = sin (zenith2)
    cos_z2 = cos (zenith2)
    cos_delta_az = cos (azimuth2 - azimuth1)
    return arccos (sin_z1 * sin_z2 * cos_delta_az + cos_z1 * cos_z2)
    # return arccos (
    #         sin (zenith1) * cos (azimuth1)  *  sin (zenith2) * cos (azimuth2)
    #         + sin (zenith1) * sin (azimuth1)  *  sin (zenith2) * sin (azimuth2)
    #         + cos (zenith1) * cos (zenith2))

def randsamp (a, count=1, seed=None):
    """Draw ``count`` values randomly from ``a``.

    :type   a: array-like
    :param  a: The array.

    :type   count: int
    :param  count: The number of values to draw.

    :return: numpy.ndarray of values.
    """
    np.random.seed (seed)
    idx = np.random.randint (0, len (a), count)
    return np.asarray (a)[idx]
