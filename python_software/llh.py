# pdf.py for grbllh

import copy
import cPickle as pickle
import logging
from itertools import izip
import datetime

import numpy as np

from _grbllh import LlhType, test_statistic
from _grbllh import Interp1DSelector, Interp1DSelectorSelector
from _grbllh import _PseudoThrower, _ProbThrower, _Thrower
from _grbllh import _TestStatDist
from _grbllh import _TestStatSF as TestStatSF

from util import _get, _timedelta_in_seconds, _prep, opening_angle
import util

from pdf import _get_duration_gauss_frac_max_pdf, _pdf_space_sig, gbm_sys
from pdf import Distribution, SeasonalVariation
from pdf import PDFSpaceBg, PDFRatioEnergy, FlatPDFRatioEnergy

pi = np.pi


def pssim_events (point, diffsim,
                  mode='circle',
                  frac_of_sphere=.05):
    """Get the idx (array of bools) for representative events.

    :type   point: mapping
    :param  point: Object with 'zenith' and 'azimuth' keys and float values.

    :type   diffsim: mapping
    :param  diffsim: Object with 'truth_zenith' and 'truth_azimuth' keys and
        ndarray values.

    :type   mode: str
    :param  mode: Either 'circle' (nearby events) or 'band' (in zenith)

    :type   frac_of_sphere: float
    :param  frac_of_sphere: Determines width of circle or zenith band.

    :return: ndarray of bools.

    Note: where a mapping is required, either subscript (object[key]) or
    attribute (getattr (object, key)) is allowed.

    """
    if mode == 'circle':
        dpsi = opening_angle (
            _get (point, 'zenith'), _get (point, 'azimuth'),
            _get (diffsim, 'true_zenith'), _get (diffsim, 'true_azimuth'))
        dpsi_thresh = np.arccos (1 - 2 * frac_of_sphere)
        idx = dpsi <= dpsi_thresh
        return idx
    elif mode == 'band':
        dcz = frac_of_sphere * 2
        cz = np.cos (_get (point, 'zenith'))
        diff_cz = np.cos (_get (diffsim, 'true_zenith'))
        diff_cz_min = diff_cz.min ()
        diff_cz_max = diff_cz.max ()
        # if cz - .5 * dcz < diff_cz_min:
        #     cz_min, cz_max = (diff_cz_min, diff_cz_min + dcz)
        if cz + .5 * dcz > diff_cz_max:
            cz_min, cz_max = (diff_cz_max - dcz, diff_cz_max)
        else:
            cz_min, cz_max = cz - .5 * dcz, cz + .5 * dcz
        idx = (cz_min <= diff_cz) * (diff_cz < cz_max)
        return idx
    else:
        raise NotImplementedError ('mode="{0}" not implemented'.format (mode))


def corrected_frac_of_sphere (frac_of_sphere, point, band_zen_range):
    """Get the corrected fraction of sphere if there is a horizon.

    :type   frac_of_sphere: float
    :param  frac_of_sphere: The original fraction of sphere (assumed to be circular)

    :type   point: mapping
    :param  point: Object with 'zenith' and 'azimuth' keys and float values.

    :type   point: list, or tuple, or array
    :param  point: Range of zenith which data is available in

    :return: float: the fraction of the sphere actually drawn from.

    """

    # following http://arxiv.org/pdf/1205.1396.pdf
    def omega (theta, t_x, t_y):
        cos_phi = (t_y * np.cos (theta)) / (t_x * np.sin (theta))
        if cos_phi > 1:
            cos_phi = 1
        elif cos_phi < -1:
            cos_phi = -1
        cos_beta = t_y / (np.sin (theta) * np.sqrt (t_x**2 + t_y**2))
        if cos_beta > 1:
            cos_beta = 1
        elif cos_beta < -1:
            cos_beta = -1
        return (2. * (np.arccos (cos_beta) - np.arccos (cos_phi) *
                        np.cos (theta)))

    theta_2 = np.arccos (1 - 2 * frac_of_sphere)
    alpha_tmp = _get (point, 'zenith')

    theta_1_tmp = np.min (band_zen_range)
    if theta_1_tmp > (np.pi / 2.):
        alpha = np.pi - alpha_tmp
        theta_1 = np.pi - theta_1_tmp
    else:
        alpha = alpha_tmp
        theta_1 = theta_1_tmp
    t_y_1 = np.cos (theta_2) - np.cos (alpha) * np.cos (theta_1)
    t_x_1 = np.sin (alpha) * np.cos (theta_1)
    t_y_2 = np.cos (theta_1) - np.cos (alpha) * np.cos (theta_2)
    t_x_2 = np.sin (alpha) * np.cos (theta_2)
    omega_1 = omega (theta_1, t_x_1, t_y_1) + omega (theta_2, t_x_2, t_y_2)
    if theta_1_tmp > (np.pi / 2.):
        omega_1 = frac_of_sphere * 4. * pi - omega_1

    theta_1_tmp = np.max (band_zen_range)
    if theta_1_tmp > (np.pi / 2.):
        alpha = np.pi - alpha_tmp
        theta_1 = np.pi - theta_1_tmp
    else:
        alpha = alpha_tmp
        theta_1 = theta_1_tmp
    t_y_1 = np.cos (theta_2) - np.cos (alpha) * np.cos (theta_1)
    t_x_1 = np.sin (alpha) * np.cos (theta_1)
    t_y_2 = np.cos (theta_1) - np.cos (alpha) * np.cos (theta_2)
    t_x_2 = np.sin (alpha) * np.cos (theta_2)
    omega_2 = omega (theta_1, t_x_1, t_y_1) + omega (theta_2, t_x_2, t_y_2)
    if theta_1_tmp > (np.pi / 2.):
        omega_2 = frac_of_sphere * 4. * pi - omega_2

    omega_tot = omega_2 - omega_1
    return (omega_tot / (4. * pi))

def observed_test_statistic (events, sources,
        pdf_space_bg, pdf_ratio_energy, source_n_b,
        config, llh_type=LlhType.per_confchan,
        full_output=False):
    """Calculate the test statistic for a given data sample.

    :type   events: :class:`Events` or list of :class:`Events`
    :param  events: The events.

    :type   sources: :class:`Sources` or list of :class:`Sources`
    :param  sources: The sources.

    :type   pdf_space_bg: :class:`PDFSpaceBg` or list of :class:`PDFSpaceBg`
    :param  pdf_space_bg: The background space PDF.

    :type   pdf_ratio_energy: :class:`PDFRatioEnergy` or list of :class:`PDFRatioEnergy`
    :param  pdf_ratio_energy: The energy PDF ratio.

    :type   source_n_b: sequence of float or list of sequences of float
    :param  source_n_b: The per-source expected numbers of background events.

    :type   config: :class:`Config`
    :param  config: The analysis configuration.

    :type   llh_type: LlhType
    :param  llh_type: One of LlhType.overall, LlhType.per_confchan,
        LlhType.per_source, LlhType.max_confchan, LlhType.max_source.

    :type   full_output: bool
    :param  full_output: If false, return T alone. If true, return a dict
        containing T, an array of events used, and a corresponding array of
        bursts the events were used for.

    """

    list_sources = util._get_list (sources)
    list_events = util._get_list (events)
    list_pres = util._get_list (pdf_ratio_energy)
    list_psbs = util._get_list (pdf_space_bg)

    if isinstance (source_n_b[0], list):
        list_source_n_b = source_n_b
    else:
        list_source_n_b = [source_n_b]

    overall_n_b = sum([sum (source_n_bs) for source_n_bs in list_source_n_b])

    doing_per_source = llh_type in (LlhType.per_source, LlhType.max_source)
    doing_per_confchan = llh_type in (LlhType.per_confchan, LlhType.max_confchan)
    doing_overall = llh_type in (LlhType.overall, )

    td = lambda s: datetime.timedelta (seconds=s)
    getdt = lambda dt: 24*3600*dt.days + dt.seconds + 1e-6*dt.microseconds
    all_sources_with_events = []
    all_events_with_sources = []
    all_r = []
    all_r_time = []
    all_r_space = []
    all_r_energy = []
    all_delang = []
    all_confchan_r = []
    all_confchan_r_time = []
    all_confchan_r_space = []
    all_confchan_r_energy = []
    all_confchan_delang = []
    all_sources_r = []
    all_sources_r_time = []
    all_sources_r_space = []
    all_sources_r_energy = []
    all_sources_delang = []
    for (ss, es, pre, psb, s_n_bs) in izip (list_sources, list_events, list_pres,
                                            list_psbs, list_source_n_b):
        sources_with_events = []
        events_with_sources = []
        confchan_r = []
        confchan_r_time = []
        confchan_r_space = []
        confchan_r_energy = []
        confchan_delang = []
        sources_r = []
        sources_r_time = []
        sources_r_space = []
        sources_r_energy = []
        sources_delang = []
        dgm = _get_duration_gauss_frac_max_pdf (ss, config)
        for m, source in enumerate (ss):
            T100 = float (source.duration)
            if T100 > config.sigma_t_max:
                sigma_t = config.sigma_t_max
            elif T100 < config.sigma_t_min:
                sigma_t = config.sigma_t_min
            else:
                sigma_t = T100
            t_min = source.t - td (config.sigma_t_truncate * sigma_t)
            t_max = source.t + td (source.duration + config.sigma_t_truncate * sigma_t)
            source_r_t = dgm[2][m]
            events_pdf_space_bg = psb (es)
            events_pdf_ratio_energy = pre (es)
            az_corrected_events = es.copy ()
            event_dts = _timedelta_in_seconds (es.t - source.t)
            #86164 is the number of seconds in a sidreal day
            az_corrected_events.azimuth = \
                    (es.azimuth - event_dts / 86164 * 2 * pi) \
                    % (2 * pi)
            source_opening_angle = source.opening_angle (
                    az_corrected_events)
            source_pdf_space_sig = source.pdf_space_sig (
                    az_corrected_events)
            source_r = []
            source_r_time = []
            source_r_space = []
            source_r_energy = []
            source_delang = []

            for i in xrange (len (es)):
                event_t = es.t[i]
                if source_opening_angle[i] > config.max_delang:
                    continue
                if event_t < t_min:
                    # too early
                    continue
                elif t_min <= event_t < source.t:
                    # early gaussian tail
                    dt = getdt (source.t - event_t)
                    this_r_t = source_r_t * np.exp (- dt**2/(2*sigma_t**2))
                elif source.t <= event_t < source.t + td (source.duration):
                    # on time
                    this_r_t = source_r_t
                elif source.t + td (source.duration) <= event_t < t_max:
                    # late gaussian tail
                    dt = getdt (event_t - (source.t + td (source.duration)))
                    this_r_t = source_r_t * np.exp (- dt**2/(2*sigma_t**2))
                elif t_max <= event_t:
                    # too late
                    continue
                this_pdf_space_sig = source.pdf_space_sig
                this_r_space = source_pdf_space_sig[i] / events_pdf_space_bg[i]
                this_r_energy = events_pdf_ratio_energy[i]
                if doing_per_source:
                    n_b = source_n_b[m]
                elif doing_overall:
                    n_b = overall_n_b
                else:
                    n_b = np.sum (source_n_b)
                this_r = 1 / n_b * this_r_t * this_r_space * this_r_energy
                this_delang = opening_angle (
                        az_corrected_events.zenith[i],
                        az_corrected_events.azimuth[i],
                        source.zenith, source.azimuth)
                if this_r >= config.min_r:
                    source_r.append (this_r)
                    source_r_time.append (this_r_t)
                    source_r_space.append (this_r_space)
                    source_r_energy.append (this_r_energy)
                    source_delang.append (this_delang)
                    sources_with_events.append (m)
                    events_with_sources.append (i)
            sources_r.append (source_r)
            sources_r_time.append (source_r_time)
            sources_r_space.append (source_r_space)
            sources_r_energy.append (source_r_energy)
            sources_delang.append (source_delang)
            confchan_r = np.r_[confchan_r, source_r]
            confchan_r_time = np.r_[confchan_r_time, source_r_time]
            confchan_r_space = np.r_[confchan_r_space, source_r_space]
            confchan_r_energy = np.r_[confchan_r_energy, source_r_energy]
            confchan_delang = np.r_[confchan_delang, source_delang]
            all_r = np.r_[all_r, source_r]
            all_r_time = np.r_[all_r_time, source_r_time]
            all_r_space = np.r_[all_r_space, source_r_space]
            all_r_energy = np.r_[all_r_energy, source_r_energy]
            all_delang = np.r_[all_delang, source_delang]

        all_sources_r.append (sources_r)
        all_sources_r_time.append (sources_r_time)
        all_sources_r_space.append (sources_r_space)
        all_sources_r_energy.append (sources_r_energy)
        all_sources_delang.append (sources_delang)
        all_confchan_r.append (confchan_r)
        all_confchan_r_time.append (confchan_r_time)
        all_confchan_r_space.append (confchan_r_space)
        all_confchan_r_energy.append (confchan_r_energy)
        all_confchan_delang.append (confchan_delang)

        all_sources_with_events.append (sources_with_events)
        all_events_with_sources.append (events_with_sources)

    if doing_per_source:
        T = test_statistic (
                [map (float, source_r) for source_r in sources_r
                 for sources_r in all_sources_r],
                llh_type)
    elif doing_per_confchan:
        T = test_statistic (
                [map (float, confchan_r) for confchan_r in all_confchan_r],
                llh_type)
    else:
        T = test_statistic (
                [map (float, all_r)],
                llh_type)
    if full_output:
        #n_s = _n_s (map (float, all_r))
        return dict (
                T=T,
                #n_s=n_s,
                sources=all_sources_with_events,
                events=all_events_with_sources,
                r=all_r,
                r_time=all_r_time,
                r_space=all_r_space,
                r_energy=all_r_energy,
                delang=all_delang,
                confchan_r=all_confchan_r,
                confchan_r_time=all_confchan_r_time,
                confchan_r_space=all_confchan_r_space,
                confchan_r_energy=all_confchan_r_energy,
                confchan_delang=all_confchan_delang,
                sources_r=all_sources_r,
                sources_r_time=all_sources_r_time,
                sources_r_space=all_sources_r_space,
                sources_r_energy=all_sources_r_energy,
                sources_delang=all_sources_delang,
                )
    else:
        return T


def observed_time_windows_max_tw (events, sources, t1s, t2s,
        pdf_space_bg, pdf_ratio_energy, source_rate,
        config, llh_type=LlhType.per_confchan,
        full_output=False):
    """Calculate the test statistic for a given data sample.

    :type   events: :class:`Events` or list of :class:`Events`
    :param  events: The events.

    :type   sources: :class:`Sources` or list of :class:`Sources`
    :param  sources: The sources.

    :type   t1s: list of doubles
    :param  t1s: one or more time window start times relative to source start times

    :type   t2s: list of doubles
    :param  t2s: one or more time window end times relative to source start times

    :type   pdf_space_bg: :class:`PDFSpaceBg` or list of :class:`PDFSpaceBg`
    :param  pdf_space_bg: The background space PDF.

    :type   pdf_ratio_energy: :class:`PDFRatioEnergy` or list of :class:`PDFRatioEnergy`
    :param  pdf_ratio_energy: The energy PDF ratio.

    :type   source_rate: sequence of float or list of sequences of float
    :param  source_rate: The per-source expected rate of background events.

    :type   config: :class:`Config`
    :param  config: The analysis configuration.

    :type   llh_type: LlhType
    :param  llh_type: One of LlhType.overall, LlhType.per_confchan,
        LlhType.per_source, LlhType.max_confchan, LlhType.max_source.

    :type   full_output: bool
    :param  full_output: If false, return T alone. If true, return a dict
        containing T, an array of events used, and a corresponding array of
        bursts the events were used for.

    """

    list_sources = util._get_list (sources)
    list_events = util._get_list (events)
    list_pres = util._get_list (pdf_ratio_energy)
    list_psbs = util._get_list (pdf_space_bg)

    #print "list_sources: ", list_sources
    #print "list_events: ", list_events
    #print "list_pres: ", list_pres

    #print "source_rate: ", source_rate
    if isinstance (source_rate[0], (np.ndarray,list)):
        list_source_rate = source_rate
    else:
        list_source_rate = [source_rate]

    print "list_source_rate: ", list_source_rate
    td = lambda s: datetime.timedelta (seconds=s)
    getdt = lambda dt: 24*3600*dt.days + dt.seconds + 1e-6*dt.microseconds

    doing_per_source = llh_type in (LlhType.per_source, LlhType.max_source)
    doing_per_confchan = llh_type in (LlhType.per_confchan, LlhType.max_confchan)
    doing_overall = llh_type in (LlhType.overall, )

    T = 0.
    all_Ts = []
    all_sources_with_events = []
    all_events_with_sources = []
    for n, (t1, t2) in enumerate (izip (t1s, t2s)):
        all_r = []
        all_r_time = []
        all_r_space = []
        all_r_energy = []
        all_delang = []
        all_confchan_r = []
        all_confchan_r_time = []
        all_confchan_r_space = []
        all_confchan_r_energy = []
        all_confchan_delang = []
        all_sources_r = []
        all_sources_r_time = []
        all_sources_r_space = []
        all_sources_r_energy = []
        all_sources_delang = []
        tw_sources_with_events = []
        tw_events_with_sources = []

        #print "================="
        #print "t2: ", t2
        #print "t1: ", t1
        if isinstance (t1, (np.ndarray,list)):
            #assert(len(t1) == len(t2)) #number of starting time windows must equal to ending time windows 
            overall_n_b = np.sum ([rate * (t22s[0] - t11s[0]) for s_rate, t22, t11 in zip(list_source_rate, t2, t1) 
                                   for rate, t22s, t11s in zip(s_rate, t22, t11)])
            # flatten the t1 and t2 lists 
            #t2 = [t22s[0] for t22 in t2 for t22s in t22]
            #t1 = [t11s[0] for t11 in t1 for t11s in t11]
            #print "t2 flattened: ", t2
            #print "t1 flattened: ", t1
        else:
            overall_n_b = np.sum ([rate * (t2 - t1) for s_rate in list_source_rate 
                                   for rate in s_rate])
            #n_src = len([rate for s_rate in list_source_rate for rate in s_rate])
            #print "n_src: ", n_src
            #make lists of time windows, each element of which is associated with one source
            t2=[[[t2] for i in range(len(list_source_rate[j]))] for j in range(len(list_source_rate))]
            t1=[[[t1] for i in range(len(list_source_rate[j]))] for j in range(len(list_source_rate))]
            #print "modified t1: ", t1
            #print "modified t2: ", t2
            
        #print "overall_n_b: ", overall_n_b

        for (ss, ts, te, es, pre, psb, s_rate) in izip (list_sources, t1, t2, list_events, list_pres,
                                                list_psbs, list_source_rate):
            
            #print "ss: ", ss
            #print "ts: ", ts
            #print "te: ", te
            #print "es: ", es
            sources_with_events = []
            events_with_sources = []
            confchan_r = []
            confchan_r_time = []
            confchan_r_space = []
            confchan_r_energy = []
            confchan_delang = []
            sources_r = []
            sources_r_time = []
            sources_r_space = []
            sources_r_energy = []
            sources_delang = []
            dgm = _get_duration_gauss_frac_max_pdf (ss, config)
            #dgm = _get_duration_gauss_frac_max_pdf (ss, cc)
            #print "dgm: ", dgm
            for m, (source, t11, t22) in enumerate (zip(ss, ts, te)):
                #print "source: ", source
                #print "m: ", m
                #print "t11: ", t11
                #print "t22: ", t22
                t_min = source.t + td (t11[0])
                t_max = source.t + td (t22[0])
                events_pdf_space_bg = psb (es)
                events_pdf_ratio_energy = pre (es)
                az_corrected_events = es.copy ()
                event_dts = _timedelta_in_seconds (es.t - source.t)
                az_corrected_events.azimuth = \
                        (es.azimuth - event_dts / 86400. * 2 * pi) \
                        % (2 * pi)
                source_opening_angle = source.opening_angle (
                        az_corrected_events)
                source_pdf_space_sig = source.pdf_space_sig (
                        az_corrected_events)

                source_r = []
                source_r_time = []
                source_r_space = []
                source_r_energy = []
                source_delang = []

                for i in xrange (len (es)):
                    event_t = es.t[i]
                    if source_opening_angle[i] > config.max_delang:
                        continue

                    if event_t < t_min:
                        # too early
                        continue
                    elif event_t >= t_max:
                        # too early
                        continue
                    this_pdf_space_sig = source.pdf_space_sig
                    this_r_space = source_pdf_space_sig[i] / events_pdf_space_bg[i]
                    this_r_energy = events_pdf_ratio_energy[i]
                    if doing_per_source:
                        n_b = s_rate[m] * (t22[0] - t11[0])
                        #print "n_b: ", n_b
                    elif doing_overall:
                        n_b = overall_n_b
                    else:
                        n_b = np.sum (s_rate * (t22[0] - t11[0]))
                    this_r = 1 / n_b * this_r_space * this_r_energy
                    #print "this_r: ", this_r
                    this_delang = opening_angle (
                            az_corrected_events.zenith[i],
                            az_corrected_events.azimuth[i],
                            source.zenith, source.azimuth)
                    if this_r >= config.min_r:
                        source_r.append (this_r)
                        source_r_space.append (this_r_space)
                        source_r_energy.append (this_r_energy)
                        source_delang.append (this_delang)
                        sources_with_events.append (m)
                        events_with_sources.append (i)

                sources_r.append (source_r)
                sources_r_time.append (source_r_time)
                sources_r_space.append (source_r_space)
                sources_r_energy.append (source_r_energy)
                sources_delang.append (source_delang)
                confchan_r = np.r_[confchan_r, source_r]
                confchan_r_time = np.r_[confchan_r_time, source_r_time]
                confchan_r_space = np.r_[confchan_r_space, source_r_space]
                confchan_r_energy = np.r_[confchan_r_energy, source_r_energy]
                confchan_delang = np.r_[confchan_delang, source_delang]
                all_r = np.r_[all_r, source_r]
                all_r_time = np.r_[all_r_time, source_r_time]
                all_r_space = np.r_[all_r_space, source_r_space]
                all_r_energy = np.r_[all_r_energy, source_r_energy]
                all_delang = np.r_[all_delang, source_delang]

            all_sources_r.append (sources_r)
            all_sources_r_time.append (sources_r_time)
            all_sources_r_space.append (sources_r_space)
            all_sources_r_energy.append (sources_r_energy)
            all_sources_delang.append (sources_delang)
            all_confchan_r.append (confchan_r)
            all_confchan_r_time.append (confchan_r_time)
            all_confchan_r_space.append (confchan_r_space)
            all_confchan_r_energy.append (confchan_r_energy)
            all_confchan_delang.append (confchan_delang)

            tw_sources_with_events.append (sources_with_events)
            tw_events_with_sources.append (events_with_sources)
        
        #print "all_sources_r: ", all_sources_r
        #print "all_r: ", all_r
        #print "all_confchan_r: ", all_confchan_r
        
        import itertools
        from itertools import chain
        a = [map (float, source_r) for sources_r in all_sources_r for source_r in sources_r]
        print "source_r: ", a
        #print "map: ", [map (float, source_r) for sources_r in all_sources_r for source_r in sources_r]
        #b = itertools.chain.from_iterable(a)
        #print "list(b): ", list(b)
        if doing_per_source:
            T_tmp = test_statistic (
                a,
                llh_type)
            #print "per-source Temp: ", T_tmp
        elif doing_per_confchan:
            T_tmp = test_statistic (
                [map (float, confchan_r) for confchan_r in all_confchan_r],
                llh_type)
        else:
            T_tmp = test_statistic (
                [map (float, all_r)],
                llh_type)
            #print "stacking Temp: ", T_tmp
        if T_tmp > T:
            T = T_tmp

        all_Ts.append (T_tmp)
        all_events_with_sources.append (tw_events_with_sources)
        all_sources_with_events.append (tw_sources_with_events)


    if full_output:
        #n_s = _n_s (map (float, all_r))
        return dict (
            T=T,
            all_Ts=all_Ts,
            sources=all_sources_with_events,
            events=all_events_with_sources,
        )
    else:
        return T


def observed_time_windows_p_tw (events, sources, tsds, t1s, t2s,
        pdf_space_bg, pdf_ratio_energy, source_rate,
        config, llh_type=LlhType.per_confchan,
        full_output=False):
    """Calculate the test statistic for a given data sample.

    :type   events: :class:`Events` or list of :class:`Events`
    :param  events: The events.

    :type   sources: :class:`Sources` or list of :class:`Sources`
    :param  sources: The sources.

    :type   tsds: list of TestStatDist
    :param  tsds: one or more TestStatDist instances (one per time window,
        in same order as time window lists)

    :type   t1s: list of doubles
    :param  t1s: one or more time window start times relative to source start times

    :type   t2s: list of doubles
    :param  t2s: one or more time window end times relative to source start times

    :type   pdf_space_bg: :class:`PDFSpaceBg` or list of :class:`PDFSpaceBg`
    :param  pdf_space_bg: The background space PDF.

    :type   pdf_ratio_energy: :class:`PDFRatioEnergy` or list of :class:`PDFRatioEnergy`
    :param  pdf_ratio_energy: The energy PDF ratio.

    :type   source_rate: sequence of float or list of sequences of float
    :param  source_rate: The per-source expected rate of background events.

    :type   config: :class:`Config`
    :param  config: The analysis configuration.

    :type   llh_type: LlhType
    :param  llh_type: One of LlhType.overall, LlhType.per_confchan,
        LlhType.per_source, LlhType.max_confchan, LlhType.max_source.

    :type   full_output: bool
    :param  full_output: If false, return T alone. If true, return a dict
        containing T, an array of events used, and a corresponding array of
        bursts the events were used for.

    """

    list_sources = util._get_list (sources)
    #print "list_sources: ", list_sources
    list_events = util._get_list (events)
    list_pres = util._get_list (pdf_ratio_energy)
    list_psbs = util._get_list (pdf_space_bg)

    if isinstance (source_rate[0], (np.ndarray,list)):
        list_source_rate = source_rate
    else:
        list_source_rate = [source_rate]

    print "list_source_rate: ", list_source_rate
    td = lambda s: datetime.timedelta (seconds=s)
    getdt = lambda dt: 24*3600*dt.days + dt.seconds + 1e-6*dt.microseconds

    doing_per_source = llh_type in (LlhType.per_source, LlhType.max_source)
    doing_per_confchan = llh_type in (LlhType.per_confchan, LlhType.max_confchan)
    doing_overall = llh_type in (LlhType.overall, )

    p = 0.
    all_Ts = []
    all_ps = []
    all_sources_with_events = []
    all_events_with_sources = []
    for n, (tsd, t1, t2) in enumerate (izip (tsds, t1s, t2s)):
        all_r = []
        all_r_time = []
        all_r_space = []
        all_r_energy = []
        all_delang = []
        all_confchan_r = []
        all_confchan_r_time = []
        all_confchan_r_space = []
        all_confchan_r_energy = []
        all_confchan_delang = []
        all_sources_r = []
        all_sources_r_time = []
        all_sources_r_space = []
        all_sources_r_energy = []
        all_sources_delang = []
        tw_sources_with_events = []
        tw_events_with_sources = []

        if isinstance (t1, (np.ndarray,list)):
            overall_n_b = np.sum ([rate * (t22s[0] - t11s[0]) for s_rate, t22, t11 in zip(list_source_rate, t2, t1)
                                   for rate, t22s, t11s in zip(s_rate, t22, t11)])
            
        else:
            overall_n_b = np.sum ([rate * (t2 - t1) for s_rate in list_source_rate
                                   for rate in s_rate])
            #make lists of time windows, each element of which is associated with one source                                      
            t2=[[[t2] for i in range(len(list_source_rate[j]))] for j in range(len(list_source_rate))]
            t1=[[[t1] for i in range(len(list_source_rate[j]))] for j in range(len(list_source_rate))]
            #print "modified t1: ", t1
            #print "modified t2: ", t2

        for (ss, ts, te, es, pre, psb, s_rate) in izip (list_sources, t1, t2, list_events, list_pres,
                                                list_psbs, list_source_rate):
            sources_with_events = []
            events_with_sources = []
            confchan_r = []
            confchan_r_time = []
            confchan_r_space = []
            confchan_r_energy = []
            confchan_delang = []
            sources_r = []
            sources_r_time = []
            sources_r_space = []
            sources_r_energy = []
            sources_delang = []
            dgm = _get_duration_gauss_frac_max_pdf (ss, config)
            for m, (source, t11, t22) in enumerate (zip(ss, ts, te)):
                t_min = source.t + td (t11[0])
                t_max = source.t + td (t22[0])
                events_pdf_space_bg = psb (es)
                events_pdf_ratio_energy = pre (es)
                az_corrected_events = es.copy ()
                event_dts = _timedelta_in_seconds (es.t - source.t)
                az_corrected_events.azimuth = \
                        (es.azimuth - event_dts / 86400. * 2 * pi) \
                        % (2 * pi)
                source_opening_angle = source.opening_angle (
                        az_corrected_events)
                source_pdf_space_sig = source.pdf_space_sig (
                        az_corrected_events)

                source_r = []
                source_r_time = []
                source_r_space = []
                source_r_energy = []
                source_delang = []

                for i in xrange (len (es)):
                    event_t = es.t[i]
                    if source_opening_angle[i] > config.max_delang:
                        continue

                    if event_t < t_min:
                        # too early
                        continue
                    elif event_t >= t_max:
                        # too early
                        continue
                    this_pdf_space_sig = source.pdf_space_sig
                    this_r_space = source_pdf_space_sig[i] / events_pdf_space_bg[i]
                    this_r_energy = events_pdf_ratio_energy[i]
                    if doing_per_source:
                        n_b = s_rate[m] * (t22[0] - t11[0])
                    elif doing_overall:
                        n_b = overall_n_b
                    else:
                        n_b = np.sum (s_rate * (t22[0] - t11[0]))
                    this_r = 1 / n_b * this_r_space * this_r_energy
                    this_delang = opening_angle (
                            az_corrected_events.zenith[i],
                            az_corrected_events.azimuth[i],
                            source.zenith, source.azimuth)
                    if this_r >= config.min_r:
                        source_r.append (this_r)
                        source_r_space.append (this_r_space)
                        source_r_energy.append (this_r_energy)
                        source_delang.append (this_delang)
                        sources_with_events.append (m)
                        events_with_sources.append (i)

                sources_r.append (source_r)
                sources_r_time.append (source_r_time)
                sources_r_space.append (source_r_space)
                sources_r_energy.append (source_r_energy)
                sources_delang.append (source_delang)
                confchan_r = np.r_[confchan_r, source_r]
                confchan_r_time = np.r_[confchan_r_time, source_r_time]
                confchan_r_space = np.r_[confchan_r_space, source_r_space]
                confchan_r_energy = np.r_[confchan_r_energy, source_r_energy]
                confchan_delang = np.r_[confchan_delang, source_delang]
                all_r = np.r_[all_r, source_r]
                all_r_time = np.r_[all_r_time, source_r_time]
                all_r_space = np.r_[all_r_space, source_r_space]
                all_r_energy = np.r_[all_r_energy, source_r_energy]
                all_delang = np.r_[all_delang, source_delang]

            all_sources_r.append (sources_r)
            all_sources_r_time.append (sources_r_time)
            all_sources_r_space.append (sources_r_space)
            all_sources_r_energy.append (sources_r_energy)
            all_sources_delang.append (sources_delang)
            all_confchan_r.append (confchan_r)
            all_confchan_r_time.append (confchan_r_time)
            all_confchan_r_space.append (confchan_r_space)
            all_confchan_r_energy.append (confchan_r_energy)
            all_confchan_delang.append (confchan_delang)

            tw_sources_with_events.append (sources_with_events)
            tw_events_with_sources.append (events_with_sources)

        if doing_per_source:
            T = test_statistic (
                    [map (float, source_r) for sources_r in all_sources_r for source_r in sources_r],
                    llh_type)
        elif doing_per_confchan:
            T = test_statistic (
                    [map (float, confchan_r) for confchan_r in all_confchan_r],
                    llh_type)
        else:
            T = test_statistic (
                    [map (float, all_r)],
                    llh_type)

        p_tmp = np.log10 (tsd.prob (T))
        if p_tmp < p:
            p = p_tmp

        all_Ts.append (T)
        all_ps.append (p_tmp)
        all_events_with_sources.append (tw_events_with_sources)
        all_sources_with_events.append (tw_sources_with_events)

        a = [map (float, source_r) for sources_r in all_sources_r for source_r in sources_r]
        print "source_r: ", a

    if full_output:
        #n_s = _n_s (map (float, all_r))
        return dict (
            p=p,
            all_Ts=all_Ts,
            all_ps=all_ps,
            sources=all_sources_with_events,
            events=all_events_with_sources,
            r_vals=a
        )
    else:
        return p


class Source (object):

    """Handle information about a single source.

    Source holds information about a single source from a :class:`Sources`
    ensemble.  This is generally a read-only class for the user: you should not
    have to construct one of these objects yourself.
    """

    def __init__ (self, zenith, azimuth, sigma, duration, t=None,
            source_index=None,
            gbm_pos=False):
        """Initialize Source with given event properties.

        :type   zenith: float
        :param  zenith: GRB zenith (radians).

        :type   azimuth: float
        :param  azimuth: GRB azimuth (radians).

        :type   sigma: float
        :param  sigma: GRB angular uncertainty (radians).

        :type   duration: float
        :param  duration: GRB start time (seconds).

        :type   t: float
        :param  t: GRB start time (datetime.datetime) (optional).
        """
        self.zenith = zenith
        self.azimuth = azimuth
        self.sigma = sigma
        self.duration = duration
        self.source_index = source_index
        self.t = t
        self.gbm_pos = gbm_pos

    def pdf_space_sig (self, events):
        """Return the value of the signal space PDF for this source and some
        Events.

        :type   events: :class:`Events`
        :param  events: An ensemble of events.

        :return: An ndarray of the PDF values.
        """
        return _pdf_space_sig (
                self.zenith, self.azimuth, self.sigma,
                events.zenith, events.azimuth, events.sigma,
                self.gbm_pos)

    def opening_angle (self, events):
        """Return the value of the opening angle between this source and some
        Events.

        :type   events: :class:`Events`
        :param  events: An ensemble of events.

        :return: An ndarray of the opening angle values in radians.
        """
        return opening_angle (
                self.zenith, self.azimuth,
                events.zenith, events.azimuth,)


class Sources (object):

    """Handle information about a set of sources.

    Sources holds the information about an ensemble of GRBs.  The information
    is stored ndarrays, but the ensemble of sources may be iterated over as
    individual :class:`Source` objects.
    """

    @staticmethod
    def random (N, sigmas, durations, upgoing=False):
        """Generate a random source catalog with N sources.

        `N` random directions will be selected; all zeniths will be greater
        than pi/2 if `upgoing`.  sigma and duration will be drawn randomly from
        `sigmas` and `durations`, which should come from past catalog GRBs.
        """
        zenith, azimuth = _get_random_directions (N, upgoing)
        sigma = sigmas[np.random.randint (0, len (sigmas), N)]
        duration = durations[np.random.randint (0, len (durations), N)]
        return Sources (zenith, azimuth, sigma, duration)

    def __init__ (self, zenith, azimuth, sigma, duration, t=None,
            source_index=None,
            gbm_pos=None,
            smear=False,
            seed=None
            ):
        """Initialize Sources with arrays of event properties.

        :type   zenith: ndarray
        :param  zenith: Per-GRB zenith (radians).

        :type   azimuth: ndarray
        :param  azimuth: Per-GRB azimuth (radians).

        :type   sigma: ndarray
        :param  sigma: Per-GRB angular uncertainty (radians).

        :type   duration: ndarray
        :param  duration: Per-GRB start time (seconds).

        :type   t: ndarray
        :param  t: Per-GRB start time (datetime.datetime objects) (optional).

        :type   source_index: ndarray of ints
        :param  source_index: The index of each source.  If not given,
            range(len(zenith)) is assumed.  Array must be increasing.

        :type   gbm_pos: ndarray of bools
        :param  gbm_pos: Per-GRB GBM localization or not

        :type   smear: bool
        :param  smear: If True, smear the source locations according to their
            given errors.

        :type   seed: int
        :param  seed: seed for true source smearing
        """
        self.zenith = zenith.copy ()
        self.orig_zenith = zenith.copy ()
        self.azimuth = azimuth.copy ()
        self.orig_azimuth = azimuth.copy ()
        self.sigma = sigma.copy ()
        self.orig_sigma = sigma.copy ()
        print "============================"
        print "self.duration: ", duration
        #self.duration = duration.copy ()
        self.duration = duration
        if t is not None:
            self.t = t.copy ()
        else:
            self.t = None
        if source_index is not None:
            self.source_index = source_index.copy ()
        else:
            self.source_index = np.arange (len (zenith))
        if gbm_pos is not None:
            self.gbm_pos = gbm_pos.copy ()
            if self.gbm_pos.dtype != 'bool':
                logging.error ('gbm_pos array is not of type bool like expected!')
        else:
            self.gbm_pos = np.zeros (len (zenith), dtype=bool)
        if smear:
            self._smear_positions (seed=seed)
        if np.sum (np.diff (self.source_index) < 0) >= 1:
            raise ValueError ('source_index must be increasing')

    def __iter__ (self):
        for i in xrange (len (self.zenith)):
            if self.t is not None:
                t = self.t[i]
            else:
                t = None
            yield Source (
                    self.zenith[i], self.azimuth[i],
                    self.sigma[i], self.duration[i], t,
                    self.source_index[i],
                    self.gbm_pos[i])

    def _smear_positions (self, seed=None):
        """ Smear detected position of sources according to their errors. """

        # save original values
        tot_sigma = self.orig_sigma.copy ()
        n = len (self.orig_zenith)

        # get total error including sys for GBM
        np.random.seed (seed)
        dice = np.random.random (n)
        gbm1_idx = self.gbm_pos * (dice < gbm_sys.weight_1)
        gbm2_idx = self.gbm_pos * (dice >= gbm_sys.weight_1)
        tot_sigma[gbm1_idx] = np.sqrt (
                self.orig_sigma[gbm1_idx]**2 + gbm_sys.sigma_1**2)
        tot_sigma[gbm2_idx] = np.sqrt (
                self.orig_sigma[gbm2_idx]**2 + gbm_sys.sigma_2**2)

        # choose random coordinates about x axis
        xaxis_azimuth = tot_sigma * np.random.randn (n)
        xaxis_zenith = pi/2 + tot_sigma * np.random.randn (n)

        # cosines and sines
        cz = np.cos (xaxis_zenith)
        sz = np.sin (xaxis_zenith)
        ca = np.cos (xaxis_azimuth)
        sa = np.sin (xaxis_azimuth)
        coz = np.cos (pi/2 - self.orig_zenith)
        soz = np.sin (pi/2 - self.orig_zenith)

        # coordinates (near x axis) and rotation matrix (about y axis)
        filler_0 = np.zeros_like (coz)
        filler_1 = np.ones_like (coz)
        xaxis_xyz = np.array ([ca*sz, sa*sz, cz])
        R_y = np.array ([
            [coz,       filler_0,   -soz,       ],
            [filler_0,  filler_1,   filler_0,   ],
            [soz,       filler_0,   coz,        ]
            ])

        # calculate R_y * xaxis_xyz manually, since axis=-1 is the
        # per-data-point running index
        x = np.sum (R_y[0, :, :] * xaxis_xyz[:, :], axis=0)
        y = np.sum (R_y[1, :, :] * xaxis_xyz[:, :], axis=0)
        z = np.sum (R_y[2, :, :] * xaxis_xyz[:, :], axis=0)

        # get the zenith and azimuth
        smeared_zenith = np.arccos (z)
        smeared_azimuth = np.arctan2 (y, x) + self.orig_azimuth
        smeared_azimuth[smeared_azimuth < 0.] += 2 * pi
        smeared_azimuth[smeared_azimuth > 2*pi] -= 2 * pi
        self.zenith, self.azimuth = smeared_zenith, smeared_azimuth

    def __len__ (self):
        return self.zenith.size

    def copy (self, idx=None):
        """Get a copy of these Sources.

        :type   idx: ndarray
        :param  idx: Array of bools specifying which events to keep.

        :return: A new :class:`Sources` instance.
        """
        if idx is None:
            idx = np.ones (len (self.zenith), dtype=bool)
        zenith = self.zenith[idx]
        azimuth = self.azimuth[idx]
        sigma = self.sigma[idx]
        duration = self.duration[idx]
        source_index = self.source_index[idx]
        gbm_pos = self.gbm_pos[idx]
        if self.t is not None:
            t = self.t[idx]
        else:
            t = self.t
        return Sources (zenith, azimuth, sigma, duration, t,
                source_index, gbm_pos)

    def cut (self, idx):
        """Apply a cut to these Sources in place.

        :type   idx: ndarray
        :param  idx: Array of bools specifying which events to keep.
        """
        self.zenith = self.zenith[idx]
        self.azimuth = self.azimuth[idx]
        self.sigma = self.sigma[idx]
        self.duration = self.duration[idx]
        self.source_index = self.source_index[idx]
        self.gbm_pos = self.gbm_pos[idx]
        if self.t is not None:
            self.t = self.t[idx]

    def pdf_space_sig (self, events):
        """Return the value of the signal space PDF for each event, for each
        source.

        :type   events: :class:`Events`
        :param  events: An ensemble of events.

        :return: A 2D ndarray of the PDF values, with shape \
                (len(self),len(sources)).
        """
        pdfs_space_sig = [source.pdf_space_sig (events) for source in self]
        return np.array (pdfs_space_sig)

    def opening_angle (self, events):
        """Return the value of the opening angle between each event and each
        source.

        :type   events: :class:`Events`
        :param  events: An ensemble of events.

        :return: A 2D ndarray of the opening angle values in radians, \
                with shape (len(self),len(sources)).
        """
        opening_angles = [source.opening_angle (events) for source in self]
        return np.array (opening_angles)


class Events (object):

    """Handle information about simulation or data events.

    Events holds the information about simulation or data events which
    is relevant to performing the unbinned likelihood analysis.  For data
    events, the livetime is required and the per-event trigger times may be
    given.  For simulation events, any signal- or background-like weights
    should be given.
    """

    def __init__ (self, zenith, azimuth, sigma, energy,
            livetime=None, livetime_per_run=None, t=None, run=None,
            weights=None):
        """Initialize an Events set with arrays of event properties.

        :type   zenith: ndarray
        :param  zenith: Per-event zenith (radians).

        :type   azimuth: ndarray
        :param  azimuth: Per-event azimuth (radians).

        :type   sigma: ndarray
        :param  sigma: Per-event angular uncertainty (radians).

        :type   energy: ndarray
        :param  energy: Per-event energy estimate (user-chosen units).

        :type   livetime: float
        :param  livetime: Ensemble livetime (for background data or simulation).

        :type   livetime_per_run: ndarray
        :param  livetime_per_run: livetime per run from data for seasonal variation modeling (dtype=float).

        :type   t: ndarray
        :param  t: Per-event start time (datetime.datetime objects).

        :type   run: ndarray
        :param  run: Per-event run number.

        :type   weights: ndarray
        :param  weights: A dictionary of weight string keys and float arrays.

        For any Events, zenith, azimuth, sigma and energy are required.  For
        data events, the livetime and, optionally, the livetime_per_run and event times should be
        given.  For simulation events, the weights should be given as a
        dictionary of weight type (string) keys to event weight (ndarray of
        floats) values.
        """
        self.zenith = zenith.copy ()
        self.azimuth = azimuth.copy ()
        self.sigma = sigma.copy ()
        self.energy = energy.copy ()
        self.livetime = livetime
        self.livetime_per_run = livetime_per_run
        self.t = None
        self.run = None
        if t is not None:
            self.t = t.copy ()
        if run is not None:
            self.run = run.copy ()
        self.weights = {}
        if weights is not None:
            for k in weights:
                self.weights[k] = weights[k].copy ()

    def __len__ (self):
        return self.zenith.size

    def copy (self, idx=None):
        """Get a copy of the Events.

        :type   idx: ndarray
        :param  idx: Array of bools specifying which events to keep.

        :return: A new :class:`Events` instance.
        """
        if idx is None:
            idx = np.ones (len (self.zenith), dtype=bool)
        zenith = self.zenith[idx]
        azimuth = self.azimuth[idx]
        sigma = self.sigma[idx]
        energy = self.energy[idx]
        livetime = self.livetime
        livetime_per_run = self.livetime_per_run
        t = None
        run = None
        if self.t is not None:
            t = self.t[idx]
        if self.run is not None:
            run = self.run[idx]
        weights = {}
        for k in self.weights:
            weights[k] = self.weights[k][idx]
        return Events (
                zenith, azimuth, sigma, energy,
                livetime, livetime_per_run, t, run, weights)

    def cut (self, idx):
        """Apply a cut to these Events in place.

        :type   idx: ndarray
        :param  idx: Array of bools specifying which events to keep.
        """

        self.zenith = self.zenith[idx]
        self.azimuth = self.azimuth[idx]
        self.sigma = self.sigma[idx]
        self.energy = self.energy[idx]
        self.livetime = self.livetime
        self.livetime_per_run = self.livetime_per_run
        if self.t is not None:
            self.t = self.t[idx]
        if self.run is not None:
            self.run = self.run[idx]
        for k in self.weights:
            self.weights[k] = self.weights[k][idx]

    def pdf_space_sig (self, event_sources):
        """Return the value of the signal space PDF for each event, given the
        source for each event.

        :type   event_sources: :class:`Sources`
        :param  event_sources: The sources corresponding to each event.

        This implementation assumes that the event sigma and source sigma
        should be added in quadrature to obtain a single 2D Gaussian width.
        """
        sz = event_sources.zenith
        sa = event_sources.azimuth
        ss = event_sources.sigma
        gbm_pos = event_sources.gbm_pos
        return _pdf_space_sig (
                sz, sa, ss, self.zenith, self.azimuth, self.sigma, gbm_pos)

    def opening_angle (self, event_sources):
        """Return the value of the opening angle between each event and the
        given source for each event.

        :type   event_sources: :class:`Sources`
        :param  event_sources: The sources corresponding to each event.
        """
        sz = event_sources.zenith
        sa = event_sources.azimuth
        return opening_angle (sz, sa, self.zenith, self.azimuth)


class TestStatDist (object):

    """Handle test statistic distributions.

    TestStatDist holds a distribution of test statistic results for pseudo
    experiments.  This class is needed because, for zero or low signal
    injection, near 90% of pseudo experiments typically give null results.
    Storing each 0 individually requires too much memory.
    """

    def __init__ (self, N_zero, T_vals, n_s_vals=None):
        """Set up a TestStatDist."""
        self._N_zero = N_zero
        self._T_vals_sorted = False
        self._T_vals = np.asarray (T_vals)
        if n_s_vals is not None:
            self._n_s_vals = np.asarray (n_s_vals)
        else:
            self._n_s_vals = None
        self._N_nonzero = len (self.T_vals)
        self._N_total = self.N_zero + self.N_nonzero

    @property
    def N_zero (self):
        """The number of zero or null results."""
        return self._N_zero

    @property
    def T_vals (self):
        """The array of non-zero test statistic values."""
        if not self._T_vals_sorted:
            if self._n_s_vals is not None:
                self._n_s_vals = self._n_s_vals[self._T_vals.argsort ()]
            self._T_vals.sort ()
            self._T_vals_sorted = True
        return self._T_vals

    @property
    def n_s_vals (self):
        """ The n_s values from a set of test statistic trials."""
        return self._n_s_vals

    @property
    def N_nonzero (self):
        """The number of non-zero or non-null results."""
        return self._N_nonzero

    @property
    def N_total (self):
        """The total number of trials included in this TestStatDist."""
        return self._N_total

    @property
    def frac_zero (self):
        """The fraction of trials with zero or null results."""
        return 1.0 * self.N_zero / self.N_total

    @property
    def frac_nonzero (self):
        """The fraction of trials with non-zero or non-null results."""
        return 1.0 * self.N_nonzero / self.N_total

    @property
    def median (self):
        i = self.N_total / 2 - self.N_zero
        if i < 0:
            return 0
        else:
            return self.T_vals[i]

    def __add__ (self, other):
        """Add two TestStatDists."""
        if self.n_s_vals is not None and other.n_s_vals is not None:
            return TestStatDist (
                self.N_zero + other.N_zero,
                np.r_[self.T_vals, other.T_vals],
                np.r_[self.n_s_vals, other.n_s_vals])
        else:
            return TestStatDist (
                self.N_zero + other.N_zero,
                np.r_[self.T_vals, other.T_vals])

    def get_hist (self, bins=100, range=None, normed=False, histlite=False):
        """Get a test statistic histogram - returns (counts, bins)."""
        if range is None:
            range = (0, self.T_vals.max ())
        h, b = np.histogram (self.T_vals, bins=bins, range=range)
        h[0] += self.N_zero
        errs = np.sqrt (h)
        if normed:
            norm = 1.0 * h.sum ()
            h = h / norm
            errs = errs / norm
        if histlite:
            from pybdt import histlite
            return histlite.Hist (b, h, errs)
        else:
            return h, b

    def prob (self, T):
        """Return the probability of getting T or higher."""
        if isinstance (T, tuple):
            if T[0] == 0:
                return 1.0
            n_bigger = (self.T_vals >= T[0]).sum ()
        else:
            if T == 0:
                return 1.0
            n_bigger = (self.T_vals >= T).sum ()
        
        p = 1.0 * n_bigger / self.N_total
        return p

    def sigma (self, T):
        """Return the number of sigmas corresponding to T or higher."""
        from scipy.stats import norm
        p = self.prob (T)
        sigma = norm.isf (p)
        return sigma

    def prob_thresh (self, p):
        """Return the test statistic threshold for `p` probability."""
        n_bigger = np.floor (p * self.N_total) + 1
        #T_vals = np.sort (self.T_vals)
        try:
            thresh = self.T_vals[-n_bigger]
            return thresh
        except:
            return 0

    def sigma_thresh (self, number_of_sigmas):
        """Return the test statistic for `number_of_sigmas` (one-sided)."""
        from scipy.stats import norm
        p = norm.sf (number_of_sigmas)
        return self.prob_thresh (p)

    @staticmethod
    def from_files (*filenames):
        """Get a TestStatDist by adding `filenames` together."""
        N_zero = 0
        T_vals_arrays = []
        n_s_vals_arrays = []
        for n, filename in enumerate (filenames):
            with open (filename) as f:
                tsd = pickle.load (f)
                N_zero += tsd.N_zero
                T_vals_arrays.append (tsd.T_vals)
                if tsd.n_s_vals:
                    n_s_vals_arrays.append (tsd.n_s_vals)
                if n % 10 == 0:
                    T_vals_arrays = [np.hstack (T_vals_arrays)]
                    n_s_vals_arrays = [np.hstack (n_s_vals_arrays)]

        # Only keep the n_s values if the arrays make sense
        if len (np.hstack (T_vals_arrays)) == len (np.hstack (n_s_vals_arrays)):
            return TestStatDist (N_zero, np.hstack (T_vals_arrays),
                                 np.hstack (n_s_vals_arrays))
        else:
            return TestStatDist (N_zero, np.hstack (T_vals_arrays))

    @property
    def _cpp_teststatdist (self):
        """Convert TestStatDist to c++ _grbllh implementation"""

        tsd = _TestStatDist (int (self.N_zero), list (self.T_vals))
        return tsd

    @property
    def cpp_teststatsf (self):
        """Convert TestStatDist to compact survival function (TestStatSF) """

        tsd = TestStatSF (self._cpp_teststatdist)
        return tsd


class Config (object):

    """Encapsulate global analysis parameters.

    This class keeps track of global paramaters including the likelihood cut,
    opening angle cut, and time PDF tail properties.
    """

    def __init__ (self,
            min_r=1.,
            max_delang=15 / 180.*pi,
            sigma_t_truncate=4,
            sigma_t_min=2.,
            sigma_t_max=30.,
            ):
        """Construct an Config instance.

        :type   min_r: float
        :param  min_r: The minimum S/(n_b * B).

        :type   max_delang: float
        :param  max_delang: The maximum angular separation of a neutrino from a
            source.

        :type   sigma_t_truncate: float
        :param  sigma_t_truncate: Number of sigmas to extend the signal time
            PDF.

        :type   sigma_t_min: float
        :param  sigma_t_min: The minimum signal time PDF tail width.

        :type   sigma_t_max: float
        :param  sigma_t_max: The maximum signal time PDF tail width.
        """
        self._sigma_t_truncate = sigma_t_truncate
        self._sigma_t_min = sigma_t_min
        self._sigma_t_max = sigma_t_max
        self._max_delang = max_delang
        self._min_r = min_r

    @property
    def sigma_t_truncate (self):
        """Number of sigmas to extend the signal time PDF."""
        return self._sigma_t_truncate
    @sigma_t_truncate.setter
    def sigma_t_truncate (self, x):
        self._sigma_t_truncate = x

    @property
    def sigma_t_min (self):
        """The minimum signal time PDF tail width."""
        return self._sigma_t_min
    @sigma_t_min.setter
    def sigma_t_min (self, x):
        self._sigma_t_min = x

    @property
    def sigma_t_max (self):
        """The maximum signal time PDF tail width."""
        return self._sigma_t_max
    @sigma_t_max.setter
    def sigma_t_max (self, x):
        self._sigma_t_max = x

    @property
    def max_delang (self):
        """The maximum angular separation of a neutrino from a source."""
        return self._max_delang
    @max_delang.setter
    def max_delang (self, x):
        self._max_delang = x

    @property
    def min_r (self):
        """The minimum S/(n_b * B)."""
        return self._min_r
    @min_r.setter
    def min_r (self, x):
        self._min_r = x


class Thrower (object):

    """Abstract base class for event throwing objects."""

    def __init__ (self):
        raise NotImplementedError ()


class PseudoBgThrower (Thrower):

    """Throw background-like pseudo-events.

    PseudoBgThrower throws background-like pseudo-events into pseudo experiments
    performed with :ref:`do_trials`.
    """

    def __init__ (self,
            confchan,
            events,
            sources,
            pdf_space_bg,
            pdf_ratio_energy,
            energy_bins=1,
            zenith_bins=1,
            config=None,
            rate_vs_time=None,
            ):
        """
        Construct a PseudoBgThrower instance.

        :type   confchan: str
        :param  confchan: The name of this configuration+channel (e.g.
            "2008_northern_tracks", "2011_cascades")

        :type   events: :class:`Events`
        :param  events: The background events.

        :type   sources: :class:`Sources`
        :param  sources: The GRBs.

        :type   pdf_space_bg: :class:`PDFSpaceBg`
        :param  pdf_space_bg: The background space PDF function object.

        :type   pdf_ratio_energy: :class:`PDFRatioEnergy`
        :param  pdf_ratio_energy: The energy PDF ratio function object.

        :type   energy_bins: int
        :param  energy_bins: The number of energy bins.

        :type   zenith_bins: int
        :param  zenith_bins: The number of zenith bins in each energy bin.

        :type   config: :class:`Config`
        :param  config: Configuration of this analysis.

        :type   rate_vs_time: class:`SeasonalVariation`
        :param  rate_vs_time: The rate as a function of time.
        """
        self.confchan = confchan
        if config is None:
            config = Config ()
        self.config = copy.deepcopy (config)

        self.pdf_space_bg = pdf_space_bg
        self.pdf_ratio_energy = pdf_ratio_energy
        self.config.energy_bins = energy_bins
        self.config.zenith_bins = zenith_bins

        # background stuff
        self.log10_energy_dist = Distribution (np.log10 (events.energy))
        self.azimuth_dist = Distribution (events.azimuth)
        logE = np.log10 (events.energy)
        h, blogE = self.log10_energy_dist.h, self.log10_energy_dist.b
        ch = 1.0 * h.cumsum () / h.sum ()
        breaks = np.linspace (0, 1, energy_bins + 1)[:-1]
        logE_breaks = list (np.r_[
            [blogE[np.where (ch > b)[0][0]]
                for b in breaks], blogE.max ()])
        logE_ranges = [
                (logE_breaks[i], logE_breaks[i+1])
                for i in xrange (len (breaks))]
        logE_idxs = [
                (logE_range[0] < logE) * (logE < logE_range[1])
                for logE_range in logE_ranges]
        cos_zenith_dists = [
                Distribution (np.cos (events.zenith[logE_idx]))
                for logE_idx in logE_idxs]
        cos_zenith_interps = [dist.icdf for dist in cos_zenith_dists]
        self.cos_zenith_selector = Interp1DSelector (
                logE_breaks, cos_zenith_interps)
        self.cos_zenith_dists = dict (izip (logE_ranges, cos_zenith_dists))

        unique_sigmas = np.unique (events.sigma)
        if len (unique_sigmas) == 1:
            self.const_sigma = unique_sigmas[0]
        else:
            sigma_selectors = []
            self.sigma_dists = {}
            for logE_range, logE_idx in izip (logE_ranges, logE_idxs):
                if logE_idx.sum () == 0:
                    raise ValueError ("Too many bins for the data. Try fewer " \
                                      "energy_bins.")
                cz = np.cos (events.zenith)
                cos_zenith_dist = Distribution (cz[logE_idx])
                s_h, s_bcz = cos_zenith_dist.h, cos_zenith_dist.b
                s_ch = 1.0 * s_h.cumsum () / s_h.sum ()
                s_breaks = np.linspace (0, 1, zenith_bins + 1)[:-1]
                cz_breaks = list (np.r_[
                    [s_bcz[np.where (s_ch > b)[0][0]]
                        for b in s_breaks], s_bcz.max ()])
                cz_ranges = [
                        (cz_breaks[i], cz_breaks[i+1])
                        for i in xrange (len (s_breaks))]
                cz_idxs = [
                        (cz_range[0] < cz) * (cz < cz_range[1])
                        for cz_range in cz_ranges]
                if 0 in [cz_idx.sum () for cz_idx in cz_idxs]:
                    raise ValueError ("Too many bins for the data. Try fewer " \
                                      "energy_bins or zenith_bins.")
                sigma_finite_idx = events.sigma < np.pi
                sigma_dists = [
                        Distribution (events.sigma[
                            logE_idx * cz_idx * sigma_finite_idx])
                        for cz_idx in cz_idxs]
                sigma_interps = [dist.icdf for dist in sigma_dists]
                sigma_selector = Interp1DSelector (
                        cz_breaks, sigma_interps)
                sigma_selectors.append (sigma_selector)
                self.sigma_dists[logE_range] = dict (
                        izip (cz_ranges, sigma_dists))
            self.sigma_selector_selector = Interp1DSelectorSelector (
                    logE_breaks, sigma_selectors)

        # sources stuff
        self.data_rate = len (events) / events.livetime
        dgm = _get_duration_gauss_frac_max_pdf (sources, config)
        self.source_duration = dgm[0]
        self.source_gauss_frac = dgm[1]
        self.source_max_pdf_ratio_time = dgm[2]
        if rate_vs_time is None:
            self.source_n_b = self.source_duration * self.data_rate
        elif sources.t is None:
            self.source_n_b = self.source_duration * rate_vs_time.best_fit['avg']
        else:
            self.source_n_b = self.source_duration \
                * rate_vs_time (sources.t)
        self.source_zenith = sources.zenith
        self.source_azimuth = sources.azimuth
        self.source_sigma = sources.sigma
        self.source_gbm_pos = sources.gbm_pos
        self.source_index = sources.source_index
        self.total_n_b = self.source_n_b.sum ()
        print( "source_n_b: ", self.source_n_b )       

    def get_copy (self, idx=None):
        """Get a copy of this thrower, possibly for a subset of sources."""
        if idx is None:
            idx = np.ones (len (self.source_index), dtype=bool)
        out = copy.deepcopy (self)
        out.source_duration = self.source_duration[idx]
        out.source_gauss_frac = self.source_gauss_frac[idx]
        out.source_max_pdf_ratio_time = self.source_max_pdf_ratio_time[idx]
        out.source_n_b = self.source_n_b[idx]
        out.source_zenith = self.source_zenith[idx]
        out.source_azimuth = self.source_azimuth[idx]
        out.source_sigma = self.source_sigma[idx]
        out.source_gbm_pos = self.source_gbm_pos[idx]
        out.source_index = self.source_index[idx]
        out.total_n_b = out.source_n_b.sum ()
        return out

    @property
    def _cpp_thrower (self):
        """Get the CPP thrower object."""
        thrower = _PseudoThrower (
                float (self.config.min_r),
                float (self.config.max_delang),
                float (self.config.sigma_t_truncate))

        if self.pdf_space_bg.use_azimuth:
            if hasattr (self, 'const_sigma'):
                thrower.set_background_const_sigma (
                        self.log10_energy_dist.icdf,
                        self.azimuth_dist.icdf,
                        self.cos_zenith_selector,
                        self.const_sigma,
                        self.pdf_space_bg.pdf,
                        self.pdf_ratio_energy.log_pdf_ratio)
            else:
                thrower.set_background (
                        self.log10_energy_dist.icdf,
                        self.azimuth_dist.icdf,
                        self.cos_zenith_selector,
                        self.sigma_selector_selector,
                        self.pdf_space_bg.pdf,
                        self.pdf_ratio_energy.log_pdf_ratio)
        else:
            if hasattr (self, 'const_sigma'):
                thrower.set_background_const_sigma (
                        self.log10_energy_dist.icdf,
                        self.cos_zenith_selector,
                        self.const_sigma,
                        self.pdf_space_bg.pdf,
                        self.pdf_ratio_energy.log_pdf_ratio)
            else:
                thrower.set_background (
                        self.log10_energy_dist.icdf,
                        self.cos_zenith_selector,
                        self.sigma_selector_selector,
                        self.pdf_space_bg.pdf,
                        self.pdf_ratio_energy.log_pdf_ratio)

        thrower.set_sources (
                _prep (self.source_index, int),
                _prep (self.source_zenith),
                _prep (self.source_azimuth),
                _prep (self.source_sigma),
                _prep (self.source_n_b / self.source_duration),
                _prep (self.source_duration),
                _prep (self.source_gauss_frac),
                _prep (self.source_max_pdf_ratio_time),
                _prep (self.source_gbm_pos, bool),
                )

        return thrower


class SignalThrower (Thrower):

    """Throw signal events."""

    def __init__ (self,
                  confchan,
                  events,
                  weight_name,
                  sources,
                  pdf_space_bg,
                  pdf_ratio_energy,
                  mu=1.,
                  t_sig_start=[],
                  t_sig_end=[],
                  config=None,
                  n_sources=None,
                  existing_thrower=None):

        """Construct a SignalThrower.

        :type   confchan: str
        :param  confchan: The name of this configuration+channel (e.g.
            "2008_northern_tracks", "2011_cascades")

        :type   events: :class:`Events`
        :param  events: The background events.

        :type   weight_name: str
        :param  weight_name: The name of the signal weight array.

        :type   sources: :class:`Sources`
        :param  sources: The GRBs.

        :type   pdf_space_bg: :class:`PDFSpaceBg`
        :param  pdf_space_bg: The background space PDF function object.

        :type   pdf_ratio_energy: :class:`PDFRatioEnergy`
        :param  pdf_ratio_energy: The energy PDF ratio function object.

        :type   mu: number
        :param  mu: Fraction by which to scale the signal weights.

        :type   config: :class:`Config`
        :param  config: Configuration of this analysis.

        :type   n_sources: int
        :param  n_sources: The number of sources.  This is needed in case the
            :class:`Sources` object is missing one or more of the last sources
            in the corresponding backgrounds due to zero simulated signal
            events surviving. For diffuse style signal injection, set to 1.

        :type   existing_thrower: :class:`SignalThrower`
        :param  existing_thrower: If specified, act as a copy constructor,
            except for applying mu if given.
        """
        if existing_thrower is not None:
            e = existing_thrower
            self.confchan = e.confchan
            self.config = e.config
            self.pdf_ratio_energy = e.pdf_ratio_energy
            self.pdf_space_bg = e.pdf_space_bg
            self.prob = mu * e.prob # important: scale by mu!
            self.r_no_n_b = e.r_no_n_b
            self.source_azimuth = e.source_azimuth
            self.source_duration = e.source_duration
            self.source_gauss_frac = e.source_gauss_frac
            self.source_max_pdf_ratio_time = e.source_max_pdf_ratio_time
            self.source_sigma = e.source_sigma
            self.source_index = e.source_index
            self.source_gbm_pos = e.source_gbm_pos
            self.source_zenith = e.source_zenith
            self.n_sources = e.n_sources
            self.t_sig_start = e.t_sig_start
            self.t_sig_end = e.t_sig_end
        else:
            if config is None:
                config = Config ()
            self.confchan = confchan
            self.config = config
            self.pdf_space_bg = pdf_space_bg
            self.pdf_ratio_energy = pdf_ratio_energy
            dgm = _get_duration_gauss_frac_max_pdf (sources, config)
            self.source_duration = dgm[0]
            self.source_gauss_frac = dgm[1]
            self.source_max_pdf_ratio_time = dgm[2]
            self.source_zenith = sources.zenith
            self.source_azimuth = sources.azimuth
            self.source_sigma = sources.sigma
            self.source_index = sources.source_index
            self.source_gbm_pos = sources.gbm_pos
            if n_sources is None:
                self.n_sources = self.source_index.max ()
            else:
                self.n_sources = n_sources
            sim_pdf_space_sig = events.pdf_space_sig (sources)
            sim_pdf_space_bg = pdf_space_bg (events)
            sim_pdf_ratio_energy = pdf_ratio_energy (events)
            all_sim_r_no_n_b = sim_pdf_space_sig \
                    * sim_pdf_ratio_energy \
                    * self.source_max_pdf_ratio_time \
                    / (sim_pdf_space_bg)
            self.r_no_n_b = all_sim_r_no_n_b
            self.prob = mu * events.weights[weight_name] # scale by mu!
            if not t_sig_start and not t_sig_end:
                self.t_sig_start = np.array (t_sig_start)
                self.t_sig_end = np.array (t_sig_end)
            else:
                self.t_sig_start = np.zeros (len (self.source_duration))
                self.t_sig_end = self.source_duration * (1 - self.source_gauss_frac)

    def __mul__ (self, mu):
        """Get a SignalThrower with probability scaled by mu."""
        return SignalThrower (None, None, None, None, None, None,
                mu=mu, existing_thrower=self)

    def __rmul__ (self, mu):
        """Get a SignalThrower with probability scaled by mu."""
        return self * mu

    @property
    def _cpp_thrower (self):
        """Get the CPP thrower object."""
        # has to be at least close to passing cut without n_b, since n_b makes
        # every r lower unless n_b < 1 (never true in stacked analyses)
        # TODO: is the above still true...?

        ##def split_arrays (x):
        ##    out = []
        ##    for m in xrange (self.n_sources):
        ##        idx = self.source_index == m
        ##        if np.sum (idx) > 0:
        ##            out.append (_prep (x[idx]))
        ##        else:
        ##            out.append ([])
        ##    return out

        source_bounds = [0]
        for m in xrange (1, self.n_sources):
            indices = np.where (self.source_index == m)[0]
            if len (indices):
                source_bounds.append (indices[0])
            else:
                source_bounds.append (source_bounds[-1])
        source_bounds.append (len (self.prob))
        thrower = _ProbThrower (
                float (self.config.min_r),
                float (self.config.sigma_t_truncate),
                _prep (source_bounds, dtype=int),
                _prep (self.source_duration),
                _prep (self.prob),
                _prep (self.r_no_n_b),
                _prep (np.zeros_like (self.source_gauss_frac)))
        return thrower


def do_trials (n_trials, bg_throwers, sig_throwers=[],
               t1=[],
               t2=[],
               llh_type=LlhType.per_confchan,
               n_sig_sources=-1,
               seed=0):
    """Perform pseudo experiments.

    :type   n_trials: number
    :param  n_trials: The number of trials to perform.

    :type   bg_throwers: list of :class:`PseudoBgThrower`
    :param  bg_throwers: One or more background Thrower instances

    :type   sig_throwers: list of :class:`SignalThrower`
    :param  sig_throwers: One or more signal Thrower instances

    :type   t1: list of doubles, or a list of lists of lists of doubles
    :param  t1: Start of multi-time window search windows, defined universally
        (one list), or per-thrower, per-source (list of lists of lists)

    :type   t2: list of doubles, or a list of lists of lists of doubles
    :param  t2: End of multi-time window search windows, defined universally
        (one list), or per-thrower, per-source (list of lists of lists)

    :type   llh_type: LlhType
    :param  llh_type: One of LlhType.overall, LlhType.per_confchan,
        LlhType.per_source, LlhType.max_confchan, LlhType.max_source.

    :type   n_sig_sources: int
    :param  n_sig_sources: Number of signal sources to select randomly for
        signal injection (default: use all sources)

    :type   seed: int
    :param  seed: The random number generator seed. (seed % 2**15 is
        actually used to avoid integer overflow.)

    :return: A TestStatDist instance.

    This function conducts pseudo experiments using the given event throwers.
    If any entries in throwers are None, they are simply ignored (this behavior
    is useful for, e.g., :ref:`find_sig_thresh` if the background is
    assumed to be negligible).
    """
    cpp_bg_throwers = [
            thrower._cpp_thrower
            for thrower in bg_throwers
            if thrower is not None]
    cpp_sig_throwers = [
            [thrower._cpp_thrower
                for thrower in sig_throwers
                if thrower is not None and thrower.confchan == bgt.confchan]
            for bgt in bg_throwers
            ]

    t1_ps = []
    t2_ps = []
    if len (t1) == 0 or not isinstance (t1[0], list):
        for thrower in bg_throwers:
            if thrower is None:
                continue
            t1_tmp = []
            t2_tmp = []
            for j in range (len (thrower.source_index)):
                t1_tmp.append (t1)
                t2_tmp.append (t2)
            t1_ps.append (t1_tmp)
            t2_ps.append (t2_tmp)
    else:
        t1_ps = t1
        t2_ps = t2

    # check that all sources have the same number of windows
    n_windows = len (t1_ps[0][0])
    for (t1_confchan, t2_confchan) in izip (t1_ps, t2_ps):
        for (t1_tmp, t2_tmp) in izip (t1_confchan, t2_confchan):
            assert (len (t1_tmp) == n_windows)
            assert (len (t2_tmp) == n_windows)

    seed = seed % 2**15
    res = _Thrower.do_trials (
            llh_type, int (n_trials),
            cpp_bg_throwers, cpp_sig_throwers, t1_ps, t2_ps,
            n_sig_sources, int (seed))
    return TestStatDist (*res)


def do_trials_pb (n_trials, bg_throwers,
                  sig_throwers=[],
                  n_sig_sources=-1,
                  seed=0):
    """Perform pseudo experiments.

    :type   n_trials: number
    :param  n_trials: The number of trials to perform.

    :type   bg_throwers: list of :class:`PseudoBgThrower`
    :param  bg_throwers: One or more background Thrower instances

    :type   sig_throwers: list of :class:`SignalThrower`
    :param  sig_throwers: One or more signal Thrower instances

    :type   n_sig_sources: int
    :param  n_sig_sources: Number of signal sources to select randomly for
        signal injection (default: use all sources)

    :type   seed: int
    :param  seed: The random number generator seed. (seed % 2**15 is
        actually used to avoid integer overflow.)

    :return: A list of lists of TestStatDist instances (one list per
        configuration/channel, one TestStatDist per-source).

    This function conducts per-burst test statistic pseudo experiments using the
    given event throwers.
    """
    cpp_bg_throwers = [
            thrower._cpp_thrower
            for thrower in bg_throwers
            if thrower is not None]
    cpp_sig_throwers = [
            [thrower._cpp_thrower
                for thrower in sig_throwers
                if thrower is not None and thrower.confchan == bgt.confchan]
            for bgt in bg_throwers
            ]

    seed = seed % 2**15
    zeross, tss, nsss = _Thrower.do_trials_pb (
            cpp_bg_throwers, cpp_sig_throwers, int (n_trials),
            n_sig_sources, int (seed))
    tsdss = [[TestStatDist (int (zero), list (t), list (ns))
             for zero, t, ns in izip (zeros, ts, nss)]
            for zeros, ts, nss in izip (zeross, tss, nsss)]

    return (tsdss)


def do_trials_ptw (n_trials, t1, t2, bg_throwers,
                   sig_throwers=[],
                   llh_type=LlhType.per_confchan,
                   n_sig_sources=-1,
                   seed=0):
    """Perform pseudo experiments.

    :type   n_trials: number
    :param  n_trials: The number of trials to perform.

    :type   t1: list of doubles, or a list of lists of lists of doubles
    :param  t1: Start of multi-time window search windows, defined universally
        (one list), or per-thrower, per-source (list of lists of lists)

    :type   t2: list of doubles, or a list of lists of lists of doubles
    :param  t2: End of multi-time window search windows, defined universally
        (one list), or per-thrower, per-source (list of lists of lists)

    :type   bg_throwers: list of :class:`PseudoBgThrower`
    :param  bg_throwers: One or more background Thrower instances

    :type   sig_throwers: list of :class:`SignalThrower`
    :param  sig_throwers: One or more signal Thrower instances

    :type   llh_type: LlhType
    :param  llh_type: One of LlhType.overall, LlhType.per_confchan,
        LlhType.per_source, LlhType.max_confchan, LlhType.max_source.

    :type   n_sig_sources: int
    :param  n_sig_sources: Number of signal sources to select randomly for
        signal injection (default: use all sources)

    :type   seed: int
    :param  seed: The random number generator seed. (seed % 2**15 is
        actually used to avoid integer overflow.)

    :return: A list of TestStatDist instances (one per-time window).
    """
    cpp_bg_throwers = [
            thrower._cpp_thrower
            for thrower in bg_throwers
            if thrower is not None]
    cpp_sig_throwers = [
            [thrower._cpp_thrower
                for thrower in sig_throwers
                if thrower is not None and thrower.confchan == bgt.confchan]
            for bgt in bg_throwers
            ]


    t1_ps = []
    t2_ps = []
    if len (t1) == 0:
        logging.error ("No windows passed. Did you mean this?")
    elif not isinstance (t1[0], list):
        for thrower in bg_throwers:
            if thrower is None:
                continue
            t1_tmp = []
            t2_tmp = []
            for j in range (len (thrower.source_index)):
                t1_tmp.append (t1)
                t2_tmp.append (t2)
            t1_ps.append (t1_tmp)
            t2_ps.append (t2_tmp)
    else:
        t1_ps = t1
        t2_ps = t2

    # check that all sources have the same number of windows
    n_windows = len (t1_ps[0][0])
    for (t1_confchan, t2_confchan) in izip (t1_ps, t2_ps):
        for (t1_tmp, t2_tmp) in izip (t1_confchan, t2_confchan):
            assert (len (t1_tmp) == n_windows)
            assert (len (t2_tmp) == n_windows)

    seed = seed % 2**15
    zeros, ts, nss = _Thrower.do_trials_ptw (
        llh_type, t1_ps, t2_ps, cpp_bg_throwers, cpp_sig_throwers,
        int (n_trials), n_sig_sources, int (seed))

    tsds = [TestStatDist (int (zero), list (t), list (ns))
            for zero, t, ns in izip (zeros, ts, nss)]

    return (tsds)


def do_trials_p_of_ps (n_trials, tsdss,
                       bg_throwers=None,
                       sig_throwers=None,
                       n_sig_sources=-1,
                       seed=0):
    """Perform pseudo experiments.

    :type   n_trials: number
    :param  n_trials: The number of trials to perform.

    :type   tsdss: list of lists of TestStatDist
    :param  tsdss: one or more lists of TestStatDist instances (one list per
        conf/channel, in same order as bg_throwers list)

    :type   bg_throwers: list of :class:`PseudoBgThrower`
    :param  bg_throwers: One or more background Thrower instances

    :type   sig_throwers: list of :class:`SignalThrower`
    :param  sig_throwers: One or more signal Thrower instances

    :type   n_sig_sources: int
    :param  n_sig_sources: Number of signal sources to select randomly for
        signal injection (default: use all sources)

    :type   seed: int
    :param  seed: The random number generator seed. (seed % 2**15 is
        actually used to avoid integer overflow.)

    :return TestStatDist of log10 (p) values

    This function conducts pseudo experiments using the given event throwers.
    If any entries in throwers are None, they are simply ignored (this behavior
    is useful for, e.g., :ref:`find_sig_thresh_p_of_ps` if the background is
    assumed to be negligible).
    """
    cpp_bg_throwers = [
            thrower._cpp_thrower
            for thrower in bg_throwers
            if thrower is not None]
    cpp_sig_throwers = [
            [thrower._cpp_thrower
                for thrower in sig_throwers
                if thrower is not None and thrower.confchan == bgt.confchan]
            for bgt in bg_throwers
            ]

    if not isinstance (tsdss[0], TestStatSF):
        try:
            cpp_tsdss = [[tsd.cpp_teststatsf for tsd in tsds]
                        for tsds in tsdss]
        except:
            cpp_tsdss = [[_TestStatSF (tsd) for tsd in tsds]
                        for tsds in tsdss]
    else:
        cpp_tsdss = tsdss

    seed = seed % 2**15
    n_ones, p_vals = _Thrower.do_trials_p_of_ps (
            cpp_tsdss, cpp_bg_throwers, cpp_sig_throwers, int (n_trials),
            n_sig_sources, int (seed))

    pvd = TestStatDist (n_ones, -np.log10 (p_vals))
    return (pvd)


def do_trials_p_of_ps_tw (n_trials, tsds, t1, t2, bg_throwers,
                          sig_throwers=[],
                          llh_type=LlhType.per_confchan,
                          n_sig_sources=-1,
                          seed=0):
    """Perform pseudo experiments.

    :type   n_trials: number
    :param  n_trials: The number of trials to perform.

    :type   tsds: list of TestStatDist
    :param  tsds: one or more TestStatDist instances (one list per time window,
        in same order as time window lists)

    :type   t1: list of doubles, or a list of lists of lists of doubles
    :param  t1: Start of multi-time window search windows, defined universally
        (one list), or per-thrower, per-source (list of lists of lists)

    :type   t2: list of doubles, or a list of lists of lists of doubles
    :param  t2: End of multi-time window search windows, defined universally
        (one list), or per-thrower, per-source (list of lists of lists)

    :type   bg_throwers: list of :class:`PseudoBgThrower`
    :param  bg_throwers: One or more background Thrower instances

    :type   sig_throwers: list of :class:`SignalThrower`
    :param  sig_throwers: One or more signal Thrower instances

    :type   llh_type: LlhType
    :param  llh_type: One of LlhType.overall, LlhType.per_confchan,
        LlhType.per_source, LlhType.max_confchan, LlhType.max_source.

    :type   n_sig_sources: int
    :param  n_sig_sources: Number of signal sources to select randomly for
        signal injection (default: use all sources)

    :type   seed: int
    :param  seed: The random number generator seed. (seed % 2**15 is
        actually used to avoid integer overflow.)

    :return: A TestStatDist of log10 (p) values

    This function conducts pseudo experiments using the given event throwers.
    If any entries in throwers are None, they are simply ignored (this behavior
    is useful for, e.g., :ref:`find_sig_thresh_p_of_ps_tw` if the background is
    assumed to be negligible).
    """
    cpp_bg_throwers = [
            thrower._cpp_thrower
            for thrower in bg_throwers
            if thrower is not None]
    cpp_sig_throwers = [
            [thrower._cpp_thrower
                for thrower in sig_throwers
                if thrower is not None and thrower.confchan == bgt.confchan]
            for bgt in bg_throwers
            ]

    t1_ps = []
    t2_ps = []
    if len (t1) == 0:
        logging.error ("No windows passed. Did you mean this?")
    elif not isinstance (t1[0], list):
        for thrower in bg_throwers:
            if thrower is None:
                continue
            t1_tmp = []
            t2_tmp = []
            for j in range (len (thrower.source_index)):
                t1_tmp.append (t1)
                t2_tmp.append (t2)
            t1_ps.append (t1_tmp)
            t2_ps.append (t2_tmp)
    else:
        t1_ps = t1
        t2_ps = t2

    # check that all sources have the same number of windows
    n_windows = len (t1_ps[0][0])
    for (t1_confchan, t2_confchan) in izip (t1_ps, t2_ps):
        for (t1_tmp, t2_tmp) in izip (t1_confchan, t2_confchan):
            assert (len (t1_tmp) == n_windows)
            assert (len (t2_tmp) == n_windows)

    if not isinstance (tsds[0], TestStatSF):
        try:
            cpp_tsds = [tsd.cpp_teststatsf for tsd in tsds]
        except:
            cpp_tsds = [TestStatSF (tsd) for tsd in tsds]
    else:
        cpp_tsds = tsds

    seed = seed % 2**15
    n_ones, p_vals = _Thrower.do_trials_p_of_ps_tw (
        llh_type, cpp_tsds, t1_ps, t2_ps, cpp_bg_throwers,
        cpp_sig_throwers, int (n_trials),
        n_sig_sources, int (seed))

    pvd = TestStatDist (n_ones, -np.log10 (p_vals))
    return (pvd)


def find_sig_thresh (thresh, beta, n_trials, bg_throwers, sig_throwers,
                     t1=[], t2=[],
                     mu_top=1., mu_bottom=0.,
                     llh_type=LlhType.per_confchan,
                     n_sig_sources=-1,
                     log=False,
                     full_output=False,
                     seed=0):
    """Find the least signal where a certain fraction of trials pass threshold.

    :type   thresh: number
    :param  thresh: The test statistic threshold.

    :type   beta: number
    :param  beta: Fraction of trials that should pass threshold.

    :type   n_trials: number
    :param  n_trials: The number of trials per normalization iteration.

    :type   bg_throwers: :class:`Thrower` or list or list of lists
    :param  bg_throwers: A background event Thrower or a list of them, or a
        list of lists of them (one list per channel).

    :type   sig_throwers: :class:`SignalThrower` or list
    :param  sig_throwers: The signal event thrower or a list of them, or a list
        of lists of them (one list per channel).

    :type   t1: list of doubles, or a list of lists of lists of doubles
    :param  t1: Start of multi-time window search windows, defined universally
        (one list), or per-thrower, per-source (list of lists of lists)

    :type   t2: list of doubles, or a list of lists of lists of doubles
    :param  t2: End of multi-time window search windows, defined universally
        (one list), or per-thrower, per-source (list of lists of lists)

    :type   mu_top: number
    :param  mu_top: Fraction of sig_thrower normalization to treat as initial
        upper bound for finding mu.

    :type   mu_bottom: number
    :param  mu_bottom: Fraction of sig_thrower normalization to treat as
        initial lower bound for finding mu.

    :type   llh_type: LlhType
    :param  llh_type: One of LlhType.overall, LlhType.per_confchan,
        LlhType.per_source, LlhType.max_confchan, LlhType.max_source.

    :type   n_sig_sources: int
    :param  n_sig_sources: Number of signal sources to select randomly for
        signal injection (default: use all sources)

    :type   log: bool
    :param  log: If true, log search information.

    :type   max_ts: bool
    :param  max_ts: Use max per-source definition of likelihood, if True;
        otherwise, use per conf/channel definition

    :type   full_output: bool
    :param  full_output: Determine the output format.

    :type   seed: int
    :param  seed: The random number generator seed.

    :return mu: If full_output is false, then the fraction of the sig_thrower
        normalization for which beta fraction of trials pass thresh.  If
        full_output is true, then a dict with mu and the TestStatDist found at
        that mu.
    """

    from scipy.optimize import brentq

    if log:
        print ('Seeking signal strength for T > {0:.3f} in'
                ' {1:.2f}% of trials'.format (thresh, 100 * beta))

    miss_vals = {}
    tsds = {}

    def miss (mu):
        try:
            miss.n_call += 1
        except:
            miss.n_call = 1
        if mu in miss_vals:
            miss_val = miss_vals[mu]
            if log:
                print ('mu {0:.4e}: '
                        'missed beta={1:.4f} by {2:.4f} (reused)'.format (
                            mu, beta, miss_val))
            return miss_val

        these_sig_throwers = [
                mu * thrower
                for thrower in sig_throwers ]
        tsd = do_trials (n_trials, bg_throwers, these_sig_throwers,
                         t1=t1, t2=t2,
                         llh_type=llh_type,
                         n_sig_sources=n_sig_sources,
                         seed=seed)
        N_disc = (tsd.T_vals > thresh).sum ()
        frac = 1.0 * N_disc / tsd.N_total
        miss_val = frac - beta
        miss_vals[mu] = miss_val
        tsds[mu] = tsd
        if log:
            print ('mu {0:.4e}: pass rate {1:.4f},'
                    ' missed beta={2:.4f} by {3:.4f}'.format (
                        mu, frac, beta, miss_val))
        return miss_val

    if mu_bottom < 0:
        raise RuntimeError ('mu_bottom is negative; something is wrong')
    while miss (mu_bottom) > 0:
        mu_top = mu_bottom
        mu_bottom *= .5

    while miss (mu_top) < 0:
        mu_bottom = mu_top
        mu_top *= 2.
        if np.isinf (mu_top):
            raise RuntimeError ('mu_top is inf; something is wrong')


    mu_beta = brentq (miss, mu_bottom, mu_top, xtol=1e-4 * mu_top)

    if log:
        print ('obtained mu = {0:.4e}'.format (mu_beta))

    if full_output:
        out = dict (mu=mu_beta, tsd=tsds[mu_beta])
        return out
    else:
        return mu_beta


def find_sig_thresh_p_of_ps (thresh, beta, n_trials, tsdss,
                             bg_throwers, sig_throwers,
                             mu_top=1., mu_bottom=0., n_sig_sources=-1,
                             log=False,
                             full_output=False,
                             seed=0):
    """Find the least signal where a certain fraction of trials pass threshold.

    :type   thresh: number
    :param  thresh: The test statistic threshold.

    :type   beta: number
    :param  beta: Fraction of trials that should pass threshold.

    :type   n_trials: number
    :param  n_trials: The number of trials per normalization iteration.

    :type   tsdss: TestStatDist or list or list of lists
    :param  tsdss: One or more background TestStatDist instances, or one or
        more lists of TestStatDist instances (one list per conf/channel)

    :type   bg_throwers: :class:`Thrower` or list or list of lists
    :param  bg_throwers: A background event Thrower or a list of them, or a
        list of lists of them (one list per channel).

    :type   sig_throwers: :class:`SignalThrower` or list
    :param  sig_throwers: The signal event thrower or a list of them, or a list
        of lists of them (one list per channel).

    :type   mu_top: number
    :param  mu_top: Fraction of sig_thrower normalization to treat as initial
        upper bound for finding mu.

    :type   mu_bottom: number
    :param  mu_bottom: Fraction of sig_thrower normalization to treat as
        initial lower bound for finding mu.

    :type   n_sig_sources: int
    :param  n_sig_sources: Number of signal sources to select randomly for
        signal injection (default: use all sources)

    :type   log: bool
    :param  log: If true, log search information.

    :type   full_output: bool
    :param  full_output: Determine the output format.

    :type   seed: int
    :param  seed: The random number generator seed.

    :return mu: If full_output is false, then the fraction of the sig_thrower
        normalization for which beta fraction of trials pass thresh.  If
        full_output is true, then a dict with mu and the TestStatDist found at
        that mu.
    """

    from scipy.optimize import brentq

    if log:
        print ('Seeking signal strength for -log (p) > {0:.3f} in'
                ' {1:.2f}% of trials'.format (thresh, 100 * beta))

    miss_vals = {}
    pvds = {}
    def get_list_of_lists (thing):
        try:
            iter (thing[0])
        except:
            try:
                iter (thing)
                thing = [thing]
            except:
                thing = [[thing]]
        return thing

    tsdss = get_list_of_lists (tsdss)
    bg_throwers = get_list_of_lists (bg_throwers)
    sig_throwers = get_list_of_lists (sig_throwers)
    if log:
        print ('TestStatDist\'s being sent to do_trials:')
        print (tsdss)
        print ('Background Throwers being sent to do_trials:')
        print (bg_throwers)
        print ('Signal Throwers being sent to do_trials:')
        print (sig_throwers)

    if not isinstance (tsdss[0], TestStatSF):
        try:
            cpp_tsdss = [[tsd.cpp_teststatsf for tsd in tsds]
                        for tsds in tsdss]
        except:
            cpp_tsdss = [[_TestStatSF (tsd) for tsd in tsds]
                        for tsds in tsdss]
    else:
        cpp_tsdss = tsdss

    def miss (mu):
        try:
            miss.n_call += 1
        except:
            miss.n_call = 1
        if mu in miss_vals:
            miss_val = miss_vals[mu]
            if log:
                print ('mu {0:.4e}: '
                        'missed beta={1:.4f} by {2:.4f} (reused)'.format (
                            mu, beta, miss_val))
            return miss_val

        these_sig_throwers = [
                [mu * thrower
                for thrower in sig_throwers_subset]
                for sig_throwers_subset in sig_throwers]
        pvd = do_trials_p_of_ps (n_trials, cpp_tsdss, bg_throwers, these_sig_throwers,
                                 n_sig_sources=n_sig_sources,
                                 seed=seed)
        N_disc = (pvd.T_vals > thresh).sum ()
        frac = 1.0 * N_disc / pvd.N_total
        miss_val = frac - beta
        miss_vals[mu] = miss_val
        pvds[mu] = pvd
        if log:
            print ('mu {0:.4e}: pass rate {1:.4f},'
                    ' missed beta={2:.4f} by {3:.4f}'.format (
                        mu, frac, beta, miss_val))
        return miss_val

    if mu_bottom < 0:
        raise RuntimeError ('mu_bottom is negative; something is wrong')
    while miss (mu_bottom) > 0:
        mu_top = mu_bottom
        mu_bottom *= .5

    while miss (mu_top) < 0:
        mu_bottom = mu_top
        mu_top *= 2.
        if np.isinf (mu_top):
            raise RuntimeError ('mu_top is inf; something is wrong')

    mu_beta = brentq (miss, mu_bottom, mu_top, xtol=1e-4 * mu_top)

    if log:
        print ('obtained mu = {0:.4e}'.format (mu_beta))

    if full_output:
        out = dict (mu=mu_beta, pvd=pvds[mu_beta])
        return out
    else:
        return mu_beta


def find_sig_thresh_p_of_ps_tw (thresh, beta, n_trials, tsds, t1, t2,
                               bg_throwers, sig_throwers,
                               llh_type=LlhType.per_confchan,
                               mu_top=1., mu_bottom=0., n_sig_sources=-1,
                               log=False,
                               full_output=False,
                               seed=0):
    """Find the least signal where a certain fraction of trials pass threshold.

    :type   thresh: number
    :param  thresh: The test statistic threshold.

    :type   beta: number
    :param  beta: Fraction of trials that should pass threshold.

    :type   n_trials: number
    :param  n_trials: The number of trials per normalization iteration.

    :type   tsds: list of TestStatDist
    :param  tsds: A list of one or more background TestStatDist (one list per time window)

    :type   t1: list of doubles
    :param  t1: Start of multi-time window search windows

    :type   t2: list of doubles
    :param  t2: End of multi-time window search windows

    :type   bg_throwers: list of :class:`Thrower`
    :param  bg_throwers: A list background event Thrower (one list per channel).

    :type   sig_throwers: list of :class:`SignalThrower`
    :param  sig_throwers: The list signal event throwers.

    :type   llh_type: LlhType
    :param  llh_type: One of LlhType.overall, LlhType.per_confchan,
        LlhType.per_source, LlhType.max_confchan, LlhType.max_source.

    :type   mu_top: number
    :param  mu_top: Fraction of sig_thrower normalization to treat as initial
        upper bound for finding mu.

    :type   mu_bottom: number
    :param  mu_bottom: Fraction of sig_thrower normalization to treat as
        initial lower bound for finding mu.

    :type   n_sig_sources: int
    :param  n_sig_sources: Number of signal sources to select randomly for
        signal injection (default: use all sources)

    :type   log: bool
    :param  log: If true, log search information.

    :type   full_output: bool
    :param  full_output: Determine the output format.

    :type   seed: int
    :param  seed: The random number generator seed.

    :return mu: If full_output is false, then the fraction of the sig_thrower
        normalization for which beta fraction of trials pass thresh.  If
        full_output is true, then a dict with mu and the TestStatDist found at
        that mu.
    """

    from scipy.optimize import brentq

    if log:
        print ('Seeking signal strength for -log (p) > {0:.3f} in'
                ' {1:.2f}% of trials'.format (thresh, 100 * beta))

    miss_vals = {}
    pvds = {}

    if log:
        print ('TestStatDist\'s being sent to do_trials:')
        print (tsds)
        print ('Background Throwers being sent to do_trials:')
        print (bg_throwers)
        print ('Signal Throwers being sent to do_trials:')
        print (sig_throwers)

    assert (len (t1) == len (t2))
    assert (len (tsds) == len (t2))

    if not isinstance (tsds[0], TestStatSF):
        try:
            cpp_tsds = [tsd.cpp_teststatsf for tsd in tsds]
        except:
            cpp_tsds = [TestStatSF (tsd) for tsd in tsds]
    else:
        cpp_tsds = tsds

    def miss (mu):
        try:
            miss.n_call += 1
        except:
            miss.n_call = 1
        if mu in miss_vals:
            miss_val = miss_vals[mu]
            if log:
                print ('mu {0:.4e}: '
                        'missed beta={1:.4f} by {2:.4f} (reused)'.format (
                            mu, beta, miss_val))
            return miss_val

        these_sig_throwers = [mu * thrower for thrower in sig_throwers]
        pvd = do_trials_p_of_ps_tw (n_trials, cpp_tsds, t1, t2, bg_throwers,
                                    these_sig_throwers,
                                    llh_type=llh_type,
                                    n_sig_sources=n_sig_sources,
                                    seed=seed)
        N_disc = (pvd.T_vals > thresh).sum ()
        frac = 1.0 * N_disc / pvd.N_total
        miss_val = frac - beta
        miss_vals[mu] = miss_val
        pvds[mu] = pvd
        if log:
            print ('mu {0:.4e}: pass rate {1:.4f},'
                    ' missed beta={2:.4f} by {3:.4f}'.format (
                        mu, frac, beta, miss_val))
        return miss_val

    if mu_bottom < 0:
        raise RuntimeError ('mu_bottom is negative; something is wrong')
    while miss (mu_bottom) > 0:
        mu_top = mu_bottom
        mu_bottom *= .5

    while miss (mu_top) < 0:
        mu_bottom = mu_top
        mu_top *= 2.
        if np.isinf (mu_top):
            raise RuntimeError ('mu_top is inf; something is wrong')

    mu_beta = brentq (miss, mu_bottom, mu_top, xtol=1e-4 * mu_top)

    if log:
        print ('obtained mu = {0:.4e}'.format (mu_beta))

    if full_output:
        out = dict (mu=mu_beta, pvd=pvds[mu_beta])
        return out
    else:
        return mu_beta


