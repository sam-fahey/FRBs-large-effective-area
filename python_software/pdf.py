# pdf.py for grbllh

from __future__ import print_function

import datetime
import copy

import numpy as np
import pickle

from _grbllh import GridInterp1D, GridInterp2D, GSLInterp1D
from _grbllh import _signal_space_pdf, gbm

from util import _prep, _timedelta_in_seconds, opening_angle

gbm_sys = gbm
_reference_time = datetime.datetime (2000, 1, 1)
pi = np.pi


def _pdf_space_sig (z1, a1, s1, z2, a2, s2, source_is_gbm):
    """Return the values of the signal space PDF.

    The source or sources are given by `z1`, `a1`, and `s1`; the events are
    given by `z2`, `a2`, `s2`, where z, a and s represent zenith, azimuth and
    sigma_space, respectively.  This utility function may be used to calculate
    the simulation per-event pdf_space_sig or the data one-source x per-event
    pdf_space_sig prior to time scrambling
    """
    d_angle = opening_angle (z1, a1, z2, a2)
    sigma_sq = s1**2 + s2**2

    @np.vectorize
    def the_signal_space_pdf (tot_sigma_sq, delang, gbm_pos):
        return _signal_space_pdf (
                float (tot_sigma_sq), float (delang), bool (gbm_pos))

    return the_signal_space_pdf (sigma_sq, d_angle, source_is_gbm)


def _get_duration_gauss_frac_max_pdf (sources, config):
    """Return the duration, gauss_frac, and max_pdf_time."""
    T100 = sources.duration
    ns = config.sigma_t_truncate

    if ns == 0.:
        duration = T100
        gauss_frac = np.zeros (len (T100))
        max_pdf_ratio_time = np.ones (len (T100))
    else:
        sigma_t = T100.copy ()
        sigma_t[sigma_t < config.sigma_t_min] = config.sigma_t_min
        sigma_t[sigma_t > config.sigma_t_max] = config.sigma_t_max
        duration = T100 + 2 * ns * sigma_t
        gauss_frac = 1. - T100 / duration
        max_pdf_ratio_time = \
                (T100 + 2 * ns * sigma_t) / (T100 + np.sqrt (2*pi) * sigma_t)

    return duration, gauss_frac, max_pdf_ratio_time


class Distribution (object):

    """Class for drawing random variables from a distribution.

    Initialize a Distribution using an array.  Then self.cdf() and
    self.icdf() are available.
    """

    def __init__ (self, array, weights=None):
        """Initialize a Distribution for given data.

        :type   array: ndarray
        :param  array: The array.

        :type   weights: ndarray
        :param  weights: The per-element weights.
        """
        self.array = np.asarray (array).copy ()
        self.h, self.b = np.histogram (self.array, bins=1000, weights=weights)
        self.x = self.b[:-1] + .5 * (self.b[1] - self.b[0])
        self.norm = 1.0 * self.h.sum ()
        cdf_x = np.r_[self.b[0], self.x, self.b[-1]]
        cdf_y = np.r_[0, self.h.cumsum () / self.norm, 1]
        icdf_y = cdf_x
        icdf_x = cdf_y


        # cdf_x = self.b
        # cdf_y = np.r_[0, cumh]

        # icdf_x = self.b[idx]
        # icdf_y = cumh[idx]

        # if icdf_x[-1] < 1:
        #     icdf_x = np.r_[icdf_x, 1]
        #     icdf_y = np.r_[icdf_y, (1 + 1e-20) * icdf_y[-1]]

        # self.cdf = Interpolator (_prep (cdf_x), _prep (cdf_y))
        # idx = np.diff (icdf_x) > 0
        # self.icdf = Interpolator (_prep (icdf_x[idx]), _prep (icdf_y[idx]))

        orig_cdf = GSLInterp1D (_prep (cdf_x), _prep (cdf_y))
        dense_x = np.linspace (cdf_x.min (), cdf_x.max (), 2000)
        dense_dx = dense_x[1] - dense_x[0]
        dense_y = orig_cdf (dense_x)
        self.cdf = GridInterp1D (
                dense_x[0], dense_dx, _prep (dense_y))

        idx = np.diff (icdf_x) > 0
        orig_icdf = GSLInterp1D (_prep (icdf_x[idx]), _prep (icdf_y[idx]));
        dense_x = np.linspace (0, 1, 2000)
        dense_dx = dense_x[1] - dense_x[0]
        dense_y = orig_icdf (dense_x)
        self.icdf = GridInterp1D (
                dense_x[0], dense_dx, _prep (dense_y))


class PDFSpaceBg (object):

    """A function-object for calculating the background space PDF of events.

    PDFSpaceBg implements a background space PDF.  It should be calculated
    based on data :class:`Events`.  At present, only the 1D PDF is implemented.
    The calculation is done by the following steps:

    -   Histogram cos(zenith)
    -   Fit the histogram with a smoothing spline
    -   Integrate along this spline to find a normalizer
    -   Fit a new spline with normalized values
    -   Densely sample the normalized spline to construct a linear
        interpolating function

    """

    def __init__ (self, events, bins=15, range=None, weight_power=1., azimuth=False):
        """Initialize a PDFSpaceBg from background events.

        :type   events: :class:`Events`
        :param  events: The data Events.

        :type   bins: int
        :param  bins: The number of bins in the initial histogram.

        :type   range: 2-tuple
        :param  range: A (cos (max zenith), cos (min zenith)) tuple.

        If `azimuth` is false, fit a InterpolatedUnivariateSpline to a
        histogram of event cos(zenith) and obtain a normalized background space
        PDF.  If `azimuth` is true, fit a RectBivariateSpline to the 2d
        histogram of event directions instead (not yet implemented).
        """
        if azimuth:
            self._init_2d (events, bins, range)
        else:
            self._init_1d (events, bins, range, weight_power=weight_power)

    def _init_2d (self, events, bins, range):

        # raise NotImplementedError (
                # 'zenith + azimuth 2d background space PDF '
                # 'not yet implemented (no 2D hist in C++ _grbllh)')

        from scipy.interpolate import RectBivariateSpline

        self.use_azimuth = True
        azimuth = events.azimuth
        zenith = events.zenith
        zen_range = (np.min (zenith), np.max (zenith))
        min_zenith, max_zenith = zen_range
        cz = np.cos (zenith)
        if range is None:
            cos_min_zenith = np.cos (min_zenith)
            cos_max_zenith = np.cos (max_zenith)
            range = (cos_max_zenith, cos_min_zenith)
        else:
            cos_max_zenith, cos_min_zenith = range

        val, aval, czval = np.histogram2d (azimuth, cz, bins=bins,
                                           range=((0, 2 * pi), range),
                                           normed=True)
        # remove last edge
        aval = aval[:-1]
        czval = czval[:-1]
        # handle periodicity
        Val = np.vstack ((val, val, val))
        Aval = np.hstack ((aval - 2 * pi, aval, aval + 2 * pi))
        # spline
        orig_B_space_func = RectBivariateSpline (Aval, czval, Val, kx=5, ky=5)
        raw_B_space_func = np.vectorize (lambda a,cz: orig_B_space_func (a,cz))
        # calculate norm
        N_bin = 500
        dense_azimuth = np.linspace (0, 2*pi, N_bin)
        dense_cos_zenith = np.linspace (cos_max_zenith, cos_min_zenith, N_bin)
        dA = dense_azimuth[1] - dense_azimuth[0]
        dCZ = dense_cos_zenith[1] - dense_cos_zenith[0]
        A, CZ = np.meshgrid (dense_azimuth, dense_cos_zenith)
        dense_pdf = raw_B_space_func (A, CZ)
        norm = np.sum (dense_pdf * dA * dCZ)
        self.raw_pdf = raw_B_space_func
        self.pdf = GridInterp2D (
            A[0][0], dA, CZ[0][0], dCZ, [_prep (row / norm) for row in dense_pdf])
        #print("\nDUMPING PICKLE PDF\n")
        #with open('/data/user/sfahey/FRB/L2_Analysis/bg_pdf_2d.pkl', 'wb') as f: pickle.dump([A, CZ, dense_pdf], f)

    def _init_1d (self, events, bins, range, weight_power=1.):
        from scipy.interpolate import UnivariateSpline
        from scipy.interpolate import InterpolatedUnivariateSpline
        self.use_azimuth = False
        zenith = events.zenith
        cz = np.cos (zenith)
        if range is None:
            range = np.min (cz), np.max (cz)
        cos_max_zenith, cos_min_zenith = range
        h, b = np.histogram (
                cz, bins=bins, range=(cos_max_zenith, cos_min_zenith))
        db = b[1] - b[0]
        x = np.r_[b[0], b[:-1] + .5 * db, b[-1]]
        y = np.r_[h[0], h, h[-1]]
        yerr = np.sqrt (y)
        raw_pdf = UnivariateSpline (x, y, (1./yerr)**weight_power)

        dense_b = np.linspace (cos_max_zenith, cos_min_zenith, 10000)
        dense_db = dense_b[1] - dense_b[0]
        dense_pdf = raw_pdf (dense_b)

        #print("\nDUMPING PICKLE PDF\n")
        #with open('/data/user/sfahey/FRB/L2_Analysis/bg_pdf.pkl', 'wb') as f: pickle.dump([dense_b, dense_pdf], f)

        self.x = x
        self.y = y
        self.yerr = yerr
        # integral over sphere should be one
        self.norm = np.sum (dense_pdf) * dense_db * 2 * pi
        self.raw_pdf = raw_pdf
        self.pdf = GridInterp1D (
                dense_b[0], dense_db, _prep (dense_pdf / self.norm))

    def __call__ (self, events):
        """Evaluate the PDFSpaceBg for some Events.

        :type   events: :class:`Events`
        :param  events: The Events.

        :return: Array of PDF values.
        """
        if not self.use_azimuth:
            return self.apply (events.zenith)
        else:
            # raise NotImplementedError ()
            return self.apply (events.zenith, events.azimuth)

    def apply (self, zenith, azimuth=None):
        """Evaluate the PDFSpaceBg for some given angles.

        :type   zenith: ndarray
        :param  zenith: Per-event zenith angles.

        :type   azimuth: ndarray
        :param  azimuth: Per-event azimuth angles.

        :return: Array of PDF values.
        """
        if not self.use_azimuth:
            if np.any (np.isnan (zenith)):
                raise ValueError ('nan in your zeniths, check your input')

            return self.pdf (np.cos (zenith))
        else:
            # raise NotImplementedError ()
            return self.pdf (azimuth, np.cos (zenith))


class PDFRatioEnergy (object):

    """A function-object for calculating the energy PDF ratios of events.

    PDFRatioEnergy implements the signal to background energy PDF ratio.  The
    signal must be taken from simulation; the background may be taken from
    the same simulation events (with different weight) or from data events.
    Normally, the former is preferable, as simulation extends to high energy.
    The calculation is done by the following steps:

    -   Histogram log10(E) for signal and background
    -   Normalize the histograms
    -   Divide the histograms
    -   Fit this ratio with a smoothing spline
    -   Densely sample the spline to construct a linear interpolating
        function

    """

    def __init__ (self,
            sim_events, sig_weight_name, bg_weight_name='',
            data_events=None, bins=30, range=None, pivot=5, **kwargs):
        """Initialize a PDFRatioEnergy.

        :type   sim_events: :class:`Events`
        :param  sim_events: Simulation events.

        :type   sig_weight_name: str
        :param  sig_weight_name: The name of the signal weight array.

        :type   bg_weight_name: str
        :param  bg_weight_name: The name of the background weight array.

        :type   data_events: :class:`Events`
        :param  data_events: Data events.

        :type   bins: int
        :param  bins: The number of bins in the initial histogram.

        :type   range: 2-tuple
        :param  range: A (min(log10(E)), max(log10(E))) tuple.

        :type   pivot: float
        :param  pivot: If using both data and simulation for the
            background PDF, ``pivot`` is the log10(E) where the background
            PDF transitions from data to simulation.

        Either ``bg_weight_name`` or ``data_events`` must be given.  If
        ``bg_weight_name`` is given, the signal and background energy PDFs will
        both come from the simulation events; otherwise, the ``data_events``
        will be used for the background energy PDF. If both are given, then
        data is used for low energies, and simulation is used past the 95th
        percentile of the data energy distribution.

        Additional kwargs are passed to the spline fit
        (scipy.interpolate.UnivariateSpline).
        """
        from scipy.interpolate import UnivariateSpline, \
                InterpolatedUnivariateSpline

        sig_logen = np.log10 (sim_events.energy)
        sig_weight = sim_events.weights[sig_weight_name]
        min_logen = sig_logen.min ()
        max_logen = sig_logen.max ()
        if bg_weight_name:
            sim_bg_logen = np.log10 (sim_events.energy)
            sim_bg_weight = sim_events.weights[bg_weight_name]
            min_logen = min (min_logen, sim_bg_logen.min ())
            max_logen = max (max_logen, sim_bg_logen.max ())
        if data_events is not None:
            data_bg_logen = np.log10 (data_events.energy)
            data_bg_weight = np.ones (len (data_events)) / data_events.livetime
            min_logen = min (min_logen, data_bg_logen.min ())
            max_logen = max (max_logen, data_bg_logen.max ())

        if range is None:
            range = (min_logen, max_logen)
        hargs = dict (bins=bins, range=range)

        sigh, b = np.histogram (sig_logen, weights=sig_weight,
                **hargs)
        errsigh, b = np.histogram (sig_logen, weights=sig_weight**2,
                **hargs)
        syerr = np.sqrt (errsigh) / sigh.sum ()
        x = b[:-1]
        self.all_x = x
        sy = sigh / sigh.sum ()
        self.sy = sy
        self.syerr = syerr

        if bg_weight_name:
            sim_bgh, b = np.histogram (
                    sim_bg_logen, weights=sim_bg_weight, **hargs)
            sim_errbgh, b = np.histogram (
                    sim_bg_logen, weights=sim_bg_weight**2, **hargs)
            sim_byerr = np.sqrt (sim_errbgh) / sim_bgh.sum ()
            sim_by = sim_bgh / sim_bgh.sum ()
            by = np.copy (sim_by)
            byerr = np.copy (sim_byerr)

        if data_events is not None:
            data_bgh, b = np.histogram (
                    data_bg_logen, weights=data_bg_weight, **hargs)
            data_errbgh, b = np.histogram (
                    data_bg_logen, weights=data_bg_weight**2, **hargs)
            data_byerr = np.sqrt (data_errbgh) / data_bgh.sum ()
            data_by = data_bgh / data_bgh.sum ()
            by = np.copy (data_by)
            byerr = np.copy (data_byerr)

        if bg_weight_name and (data_events is not None):
            i_max = np.where ((data_by > 0) * (x < pivot))[0][-1]
            left = np.arange (len (data_by)) <= i_max
            #left = (data_bgh > 0)
            #for i in xrange (-5, 0):
            #    left[left == True][i] == False
            right = ~left
            i_min = np.argmax (data_bgh)
            self.pivot = x[i_max]
            off_factor = lambda i: data_by[i] / sim_by[i]
            slope = (off_factor (i_max) - off_factor (i_min)) / (i_max - i_min)
            cor = np.vectorize (
                    lambda i: off_factor (i_max) + slope * (i - i_max) )
            corrections = np.array ([
                cor (i) for i in np.where (right)[0] ])
            by[right] = corrections * sim_by[right]
            byerr[right] = corrections * sim_byerr[right]
            norm = by.sum ()
            by /= norm
            byerr /= norm
            self.sim_by = sim_by
            self.sim_byerr = sim_byerr
            self.data_by = data_by
            self.data_byerr = data_byerr

        self.by = by
        self.byerr = byerr

        ratio = sy / by
        ratio_err = ratio * np.sqrt ((syerr/sy)**2 + (byerr/by)**2)

        idx = np.isfinite (ratio) * np.isfinite (ratio_err)

        log_ratio_spline = UnivariateSpline (
            x[idx], np.log10 (ratio[idx]), w=(ratio/ratio_err)[idx],
            **kwargs)
        #log_ratio_spline = InterpolatedUnivariateSpline (
                #x[idx], np.log10 (ratio[idx]))

        dense_x = np.linspace (x.min (), x.max (), 1000)
        dense_dx = dense_x[1] - dense_x[0]
        dense_y = log_ratio_spline (dense_x)

        # self.log_pdf_ratio = Interpolator (_prep (dense_x), _prep (dense_y))
        self.log_pdf_ratio = GridInterp1D (
                dense_x[0], dense_dx, _prep (dense_y))
        self.min = max (range[0], x[idx].min ())
        self.max = min (range[1], x[idx].max ())
        self.x = x[idx]
        self.ratio = ratio[idx]
        self.ratio_err = ratio_err[idx]
        self.idx = idx


    def __call__ (self, events):
        """Evaluate the PDFRatioEnergy for some Events.

        :type   events: :class:`Events`
        :param  events: The Events.

        :return: Array of PDF ratio values.
        """
        return self.apply (events.energy)

    def apply (self, energy):
        """Evaluate the PDFRatioEnergy for some given energies.

        :type   energy: ndarray
        :param  energy: Per-event energy.

        :return: Array of PDF values.
        """
        if np.any (np.isnan (energy)):
            raise ValueError ('nan in your energies, check your input')

        logen = np.log10 (energy)
        logen[logen < self.min] = self.min
        logen[logen > self.max] = self.max
        return 10**self.log_pdf_ratio (logen)


class FlatPDFRatioEnergy (object):

    """PDFRatioEnergy that returns ratio=1 for all energies."""

    def __init__ (self):
        self.log_pdf_ratio = GridInterp1D (
                0, 10, [0, 0])

    def __call__ (self, events):
        """Evaluate the PDFRatioEnergy for some Events.

        :type   events: :class:`Events`
        :param  events: The Events.

        :return: Array of PDF ratio values.
        """
        return self.apply (events.energy)

    def apply (self, energy):
        """Evaluate the PDFRatioEnergy for some given energies.

        :type   energy: ndarray
        :param  energy: Per-event energy.

        :return: Array of PDF values.
        """
        return np.ones_like (energy)


class PDFRatioScores (object):

    """A function-object for calculating the BDT score  PDF ratios of events.

    PDFRatioScores implements the signal to background BDT score PDF ratio.  The
    signal must be taken from simulation; the background may be taken from
    the same simulation events (with different weight) or from data events.
    Normally, the former is preferable, as simulation extends to high energy.
    The calculation is done by the following steps:

    -   Histogram scores for signal and background
    -   Normalize the histograms
    -   Divide the histograms
    -   Fit this ratio with a smoothing spline
    -   Densely sample the spline to construct a linear interpolating
        function

    """

    def __init__ (self,
            sim_events, sig_weight_name, bg_weight_name='',
            data_events=None, bins=30, range=None, pivot=5):
        """Initialize a PDFRatioEnergy.

        :type   sim_events: :class:`Events`
        :param  sim_events: Simulation events.

        :type   sig_weight_name: str
        :param  sig_weight_name: The name of the signal weight array.

        :type   bg_weight_name: str
        :param  bg_weight_name: The name of the background weight array.

        :type   data_events: :class:`Events`
        :param  data_events: Data events.

        :type   bins: int
        :param  bins: The number of bins in the initial histogram.

        :type   range: 2-tuple
        :param  range: A (min(scores), max(scores)) tuple.

        :type   pivot: float
        :param  pivot: If using both data and simulation for the
            background PDF, ``pivot`` is the score where the background
            PDF transitions from data to simulation.

        Either ``bg_weight_name`` or ``data_events`` must be given.  If
        ``bg_weight_name`` is given, the signal and background score PDFs will
        both come from the simulation events; otherwise, the ``data_events``
        will be used for the background score PDF. If both are given, then
        data is used for low energies, and simulation is used past the 95th
        percentile of the data energy distribution.
        """
        from scipy.interpolate import UnivariateSpline, \
                InterpolatedUnivariateSpline

        sig_scores = sim_events.scores
        sig_weight = sim_events.weights[sig_weight_name]
        min_score = sig_scores.min ()
        max_score = sig_scores.max ()
        if bg_weight_name:
            sim_bg_scores = sim_events.scores
            sim_bg_weight = sim_events.weights[bg_weight_name]
            min_score = min (min_score, sim_bg_score.min ())
            max_score = max (max_score, sim_bg_score.max ())
        if data_events is not None:
            data_bg_scores = data_events.scores
            data_bg_weight = np.ones (len (data_events)) / data_events.livetime
            min_score = min (min_score, data_bg_scores.min ())
            max_score = max (max_score, data_bg_scores.max ())

        if range is None:
            range = (min_score, max_score)
        hargs = dict (bins=bins, range=range)

        sigh, b = np.histogram (sig_scores, weights=sig_weight,
                **hargs)
        errsigh, b = np.histogram (sig_scores, weights=sig_weight**2,
                **hargs)
        syerr = np.sqrt (errsigh) / sigh.sum ()
        x = b[:-1]
        self.all_x = x
        sy = sigh / sigh.sum ()
        self.sy = sy
        self.syerr = syerr

        if bg_weight_name:
            sim_bgh, b = np.histogram (
                    sim_bg_scores, weights=sim_bg_weight, **hargs)
            sim_errbgh, b = np.histogram (
                    sim_bg_scores, weights=sim_bg_weight**2, **hargs)
            sim_byerr = np.sqrt (sim_errbgh) / sim_bgh.sum ()
            sim_by = sim_bgh / sim_bgh.sum ()
            by = np.copy (sim_by)
            byerr = np.copy (sim_byerr)

        if data_events is not None:
            data_bgh, b = np.histogram (
                    data_bg_scores, weights=data_bg_weight, **hargs)
            data_errbgh, b = np.histogram (
                    data_bg_scores, weights=data_bg_weight**2, **hargs)
            data_byerr = np.sqrt (data_errbgh) / data_bgh.sum ()
            data_by = data_bgh / data_bgh.sum ()
            by = np.copy (data_by)
            byerr = np.copy (data_byerr)

        if bg_weight_name and (data_events is not None):
            i_max = np.where ((data_by > 0) * (x < pivot))[0][-1]
            left = np.arange (len (data_by)) <= i_max
            #left = (data_bgh > 0)
            #for i in xrange (-5, 0):
            #    left[left == True][i] == False
            right = ~left
            i_min = np.argmax (data_bgh)
            self.pivot = x[i_max]
            off_factor = lambda i: data_by[i] / sim_by[i]
            slope = (off_factor (i_max) - off_factor (i_min)) / (i_max - i_min)
            cor = np.vectorize (
                    lambda i: off_factor (i_max) + slope * (i - i_max) )
            corrections = np.array ([
                cor (i) for i in np.where (right)[0] ])
            by[right] = corrections * sim_by[right]
            byerr[right] = corrections * sim_byerr[right]
            norm = by.sum ()
            by /= norm
            byerr /= norm
            self.sim_by = sim_by
            self.sim_byerr = sim_byerr
            self.data_by = data_by
            self.data_byerr = data_byerr

        self.by = by
        self.byerr = byerr

        ratio = sy / by
        ratio_err = ratio * np.sqrt ((syerr/sy)**2 + (byerr/by)**2)

        idx = np.isfinite (ratio) * np.isfinite (ratio_err)

        log_ratio_spline = UnivariateSpline (
                x[idx], np.log10 (ratio[idx]), w=(ratio/ratio_err)[idx])
        #log_ratio_spline = InterpolatedUnivariateSpline (
                #x[idx], np.log10 (ratio[idx]))

        dense_x = np.linspace (x.min (), x.max (), 1000)
        dense_dx = dense_x[1] - dense_x[0]
        dense_y = log_ratio_spline (dense_x)

        # self.log_pdf_ratio = Interpolator (_prep (dense_x), _prep (dense_y))
        self.log_pdf_ratio = GridInterp1D (
                dense_x[0], dense_dx, _prep (dense_y))
        self.min = max (range[0], x[idx].min ())
        self.max = min (range[1], x[idx].max ())
        self.x = x[idx]
        self.ratio = ratio[idx]
        self.ratio_err = ratio_err[idx]
        self.idx = idx


    def __call__ (self, events):
        """Evaluate the PDFRatioEnergy for some Events.

        :type   events: :class:`Events`
        :param  events: The Events.

        :return: Array of PDF ratio values.
        """
        return self.apply (events.energy)

    def apply (self, scores):
        """Evaluate the PDFRatioEnergy for some given energies.

        :type   energy: ndarray
        :param  energy: Per-event energy.

        :return: Array of PDF values.
        """
        if np.any (np.isnan (scores)):
            raise ValueError ('nan in your scores, check your input')

        scores[scores < self.min] = self.min
        scores[scores > self.max] = self.max
        return 10**self.log_pdf_ratio (scores)


class SeasonalVariation (object):

    """A function-object for calculating the bg rate as a function of time."""

    def __init__ (self, runs, livetime_per_run, times,
                  grb_runs=None,
                  duration_min=0.,
                  width_n_sigmas=1.,
                  factor=1,
                  averaged_time='per_run'):
        """Construct a :class:`SeasonalVariation` object.

        :type   runs: sequence of ints
        :param  runs: the event run numbers

        :type   livetime_per_run: sequence of floats
        :param  livetime_per_run: the true livetime per run from IceCube Live database.

        :type   times: sequence of datetime.datetime objects
        :param  times: the event trigger times

        :type   width_n_sigmas: float
        :param  width_n_sigmas: the number of standard deviations from the mean
            rate to allow in the rate vs time fit

        :type   factor: float
        :param  factor: return factor * the otherwise best fit rate vs time

        :type   averaged_time: string
        :param  averaged_time: an option to choose between 'per_run' or 'per_month' to calculate
                avergaed event rates before the seasonal variation fit.
        """
        from scipy.interpolate import UnivariateSpline
        from icecube.grbllh.fitting import curve_fit
        import numpy.ma as ma
        try:
            from itertools import izip
        except ImportError:  #python3.x
            izip = zip

        time_sort_idx = times.argsort ()
        times = times[time_sort_idx]
        runs = runs[time_sort_idx]
        livetime_per_run = livetime_per_run[time_sort_idx]
        start_indices = np.r_[0, np.nonzero (np.diff (runs))[0] + 1]
        # if start_indices[-1] == len (runs) - 2:
            # start_indices = np.r_[start_indices, len (runs) - 1]
        end_indices = np.r_[start_indices[1:] - 1, len (runs) - 1]
        run_unique = runs[start_indices]
        run_starts = times[start_indices]
        run_ends = times[end_indices]
        run_counts = (end_indices - start_indices) + 1
        run_durations = livetime_per_run[start_indices]
        run_rates = run_counts / run_durations
        run_rate_errs = np.sqrt (run_counts) / run_durations
        idx_neg_rates = 0 < run_rates
        #idx_long_run = run_durations > duration_min  # only use runs longer than an hour
        idx_long_run = run_durations > 0.  # SAM CHANGED THIS IN HIS SRC
        idx_nogrbs = np.ones (len (run_unique), dtype=bool)
        if grb_runs is not None:
            for r in grb_runs:
                idx_nogrbs -= (run_unique == r)
        idx_good_runs = idx_neg_rates * idx_long_run * idx_nogrbs

        #print ("len of idx_good_runs: ", len(idx_good_runs))
        #print ("any good runs? ", idx_good_runs.any())

        if averaged_time == 'per_run':
            print ("You are using per run rates in the seasonal variation fit... ")
            idx_nogrbs = np.ones (len (run_unique), dtype=bool)
            if grb_runs is not None:
                for r in grb_runs:
                    idx_nogrbs -= (run_unique == r)
            idx_neg_rates = 0 < run_rates
            #idx_long_run = run_durations > duration_min  # only use runs longer than an hour
            idx_long_run = run_durations > 0.  # SAM CHANGED THIS IN HIS SRC
            rate_mean = run_rates[idx_neg_rates * idx_nogrbs * idx_long_run].mean ()
            rate_std = run_rates[idx_neg_rates * idx_nogrbs * idx_long_run].std ()
            
            run_offsets = _timedelta_in_seconds (run_starts - run_starts[0])

            self._best_fit = curve_fit (self.rate_vs_offset,
                                        run_offsets[idx_neg_rates * idx_nogrbs * idx_long_run],
                                        run_rates[idx_neg_rates * idx_nogrbs * idx_long_run],
                                        p0=[.1 * rate_std, 0., rate_mean])

            self.t0 = run_starts[0]
            self.run_durations = run_durations
            self.run_rates = run_rates
            self.run_rate_errs = run_rate_errs
            self.run_starts = run_starts
            self.idx = idx_neg_rates * idx_nogrbs * idx_long_run
            self.factor = factor
            self.chisq2 = np.sum ([(rate - self.__call__ (start))**2 / rate_err**2
                                   for rate, rate_err, start in izip (
                        self.run_rates[self.idx],
                        self.run_rate_errs[self.idx],
                        self.run_starts[self.idx])])
            self.ndof = self.idx.sum () - 3

        if averaged_time == 'per_month':
            print ("You are using monthly averaged rates in the seasonal variation fit... ")
            #month = 365.25 * 86400 / 12
            #n_months = int(_timedelta_in_seconds(run_ends[-1] - run_starts[0]) / month) + 1
            month=_timedelta_in_seconds(run_ends[-1] - run_starts[0]) / 12
            n_months = int(_timedelta_in_seconds(run_ends[-1] - run_starts[0]) / month)
            month_durations = np.zeros(n_months)
            month_counts = np.zeros(n_months)
            month_mask = np.empty([n_months], dtype=np.ndarray)
            month_times = np.array([run_starts[0] for i in xrange(n_months)])
            month_offsets = np.zeros(n_months)
            for i in range(n_months):
                mask1 = ma.getmask(ma.masked_greater_equal(_timedelta_in_seconds(run_starts - run_starts[0]), i*month))
                mask2 = ma.getmask(ma.masked_less(_timedelta_in_seconds(run_starts - run_starts[0]), (i+1)*month))
                month_mask[i] = mask1 * mask2
                #print ("month_mask: ", month_mask[i])
                #print ("len of month_mask: ", len(month_mask[i]))
                #print ("any good runs? ", (idx_good_runs * month_mask[i]).any())
                #month_durations[i] = np.sum([run_duration for run_duration in run_durations[idx_good_runs * month_mask[i]]])
                #month_counts[i] = np.sum([run_count for run_count in run_counts[idx_good_runs * month_mask[i]]])
                month_durations[i] = np.sum([run_duration for run_duration in run_durations[month_mask[i]]])
                month_counts[i] = np.sum([run_count for run_count in run_counts[month_mask[i]]])
                #print "i*month/2: ", i*month/2
                #month_offsets[i] = run_starts[0] + datetime.timedelta(seconds = month/2) + datetime.timedelta(seconds = i * month)
                if i==0:
                    month_times[0] = run_starts[0] + datetime.timedelta(seconds =0.5 *  _timedelta_in_seconds(run_starts[month_mask[0]][-1] - run_starts[month_mask[0]][0]))
                    month_offsets[0] = 0.5 *  _timedelta_in_seconds(run_starts[month_mask[0]][-1] - run_starts[0])
                else:
                    month_offsets[i] = _timedelta_in_seconds(run_ends[month_mask[i-1]][-1] + datetime.timedelta(seconds = 0.5 * _timedelta_in_seconds(run_starts[month_mask[i]][-1] - run_starts[month_mask[i]][0])) - run_starts[0])
                    month_times[i] = run_ends[month_mask[i-1]][-1] + datetime.timedelta(seconds = 0.5 * _timedelta_in_seconds(run_starts[month_mask[i]][-1] - run_starts[month_mask[i]][0]))

            month_rates = month_counts / month_durations
            month_rate_errs = np.sqrt (month_counts) / month_durations

            mrate_mean = month_rates.mean ()
            mrate_std = month_rates.std ()
            mmin_rate = mrate_mean - width_n_sigmas * mrate_std
            mmax_rate = mrate_mean + width_n_sigmas * mrate_std


            self._best_fit = curve_fit (self.rate_vs_offset,
                                        month_offsets,
                                        month_rates,
                                        p0=[.1 * mrate_std, 0, mrate_mean])

            self.t0 = run_starts[0]
            self.month_durations = month_durations
            self.month_rates = month_rates
            self.month_rate_errs = month_rate_errs
            self.month_offsets = month_offsets
            self.month_times = month_times
            self.mmin_rate = mmin_rate
            self.mmax_rate = mmax_rate
            self.factor = factor
            self.chisq2 = np.sum ([(rate - self.__call__ (start))**2 / rate_err**2
                                   for rate, rate_err, start in izip (
                        self.month_rates,
                        self.month_rate_errs,
                        self.month_times)])
            self.ndof = n_months - 3


    def __call__ (self, times):
        offsets = _timedelta_in_seconds (times - self.t0)
        return self.factor * self.rate_vs_offset (offsets, *self._best_fit[0])
        #rate_vs_time = np.vectorize (lambda times:
        #        self.rate_vs_offset (_timedelta_in_seconds (times - self.t0)))
        #return rate_vs_time (times)

    def __mul__ (self, factor):
        out = copy.deepcopy (self)
        out.factor *= factor
        return out

    def __rmul__ (self, factor):
        return self * factor

    @property
    def best_fit (self):
        fit = {'amp': self._best_fit[0][0], 'phase': self._best_fit[0][1],
               'avg': self._best_fit[0][2]}
        return (fit)

    @staticmethod
    def rate_vs_offset (offset, amp, phase, avg):
        year = 365.25 * 86400
        return amp * np.sin (2 * pi / year * offset + phase) + avg

