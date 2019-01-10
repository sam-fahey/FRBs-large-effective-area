# __init__.py for grbllh

# TODO: write a test that verifies the diffsim and pssim norm conventions at
# each point along the analysis pipeline

from __future__ import print_function

import copy
import logging
from itertools import izip, chain

import numpy as np

import _grbllh
from _grbllh import LlhType
from _grbllh import GridInterp1D, GridInterp2D, Interp1DSelector, Interp1DSelectorSelector

import util
from util import _get

import pdf
from pdf import Distribution, SeasonalVariation
from pdf import PDFSpaceBg, PDFRatioEnergy, FlatPDFRatioEnergy

import llh
from llh import Source, Sources, Events
from llh import Config, Thrower, PseudoBgThrower, SignalThrower, TestStatDist

import cache


def do_trials (n_trials, analyses,
               t1=[],
               t2=[],
               llh_type=LlhType.per_confchan,
               mu=0.,
               use_diffsig=False,
               n_sig_sources=-1,
               seed=0):

    """
    Wrapper for :ref:`llh.do_trials` for use with a set of AutoAnalysis instances

    :type   n_trials: int
    :param  n_trials: Number of trials to perform

    :type   analyses: list of AutoAnalysis
    :param  analyses: A list of AutoAnalysis instances to perform pseudo-trials for

    :type   t1: list of doubles, or a list of lists of lists of doubles
    :param  t1: Start of multi-time window search windows, defined universally
        (one list), or per-thrower, per-source (list of lists of lists)

    :type   t2: list of doubles, or a list of lists of lists of doubles
    :param  t2: End of multi-time window search windows, defined universally
        (one list), or per-thrower, per-source (list of lists of lists)

    :type   llh_type: LlhType
    :param  llh_type: One of LlhType.overall, LlhType.per_confchan,
        LlhType.per_source, LlhType.max_confchan, LlhType.max_source.

    :type   mu: number
    :param  mu: Fraction by which to scale the signal weights. `mu = 0.` is the
        default, performing background-only trials.

    :type   use_diffsig: bool
    :param  use_diffsig: If True, use diffuse signal throwers.

    :type   n_sig_sources: int
    :param  n_sig_sources: Number of signal sources to select randomly for
        signal injection (default: use all sources)

    :type   seed: int
    :param  seed: The random number generator seed. (seed % 2**15 is
        actually used to avoid integer overflow.)

    :return: A TestStatDist instance.

    This function conducts pseudo experiments using the given set of
    AutoAnalysis instances.  If any entries in throwers are None, they are
    simply ignored (this behavior is useful for, e.g., :ref:`find_sig_thresh`
    if the background is assumed to be negligible).
    """

    bg_throwers = [analysis.bg_thrower for analysis in analyses]

    if mu == 0.:
        sig_throwers = []
    elif use_diffsig:
        sig_throwers = list (chain.from_iterable ([
            analysis.diffsig_throwers for analysis in analyses]))
        sig_throwers = [mu*thrower for thrower in sig_throwers]
    else:
        sig_throwers = list (chain.from_iterable ([
            analysis.pssig_throwers for analysis in analyses]))
        sig_throwers = [mu*thrower for thrower in sig_throwers]

    return (llh.do_trials (n_trials, bg_throwers, sig_throwers=sig_throwers,
                           t1=t1,
                           t2=t2,
                           llh_type=llh_type,
                           n_sig_sources=n_sig_sources,
                           seed=seed))

def do_trials_pb (n_trials, analyses,
                  mu=0.,
                  use_diffsig=False,
                  n_sig_sources=-1,
                  seed=0):

    """
    Wrapper for :ref:`llh.do_trials_pb` for use with a set of AutoAnalysis instances

    :type   n_trials: int
    :param  n_trials: Number of trials to perform

    :type   analyses: list of AutoAnalysis
    :param  analyses: A list of AutoAnalysis instances to perform pseudo-trials for

    :type   mu: number
    :param  mu: Fraction by which to scale the signal weights. `mu = 0.` is the
        default, performing background-only trials.

    :type   use_diffsig: bool
    :param  use_diffsig: If True, use diffuse signal throwers.

    :type   n_sig_sources: int
    :param  n_sig_sources: Number of signal sources to select randomly for
        signal injection (default: use all sources)

    :type   seed: int
    :param  seed: The random number generator seed. (seed % 2**15 is
        actually used to avoid integer overflow.)

    :return: A TestStatDist instance.

    This function conducts pseudo experiments using the given set of
    AutoAnalysis instances.  If any entries in throwers are None, they are
    simply ignored (this behavior is useful for, e.g., :ref:`find_sig_thresh`
    if the background is assumed to be negligible).
    """

    bg_throwers = [analysis.bg_thrower for analysis in analyses]

    if mu == 0.:
        sig_throwers = []
    elif use_diffsig:
        sig_throwers = list (chain.from_iterable ([
            analysis.diffsig_throwers for analysis in analyses]))
        sig_throwers = [mu*thrower for thrower in sig_throwers]
    else:
        sig_throwers = list (chain.from_iterable ([
            analysis.pssig_throwers for analysis in analyses]))
        sig_throwers = [mu*thrower for thrower in sig_throwers]

    return (llh.do_trials_pb (n_trials, bg_throwers, sig_throwers=sig_throwers,
                              n_sig_sources=n_sig_sources,
                              seed=seed))

def tsds_from_files_pb (filenames):
    tsdss = cache.load (filenames[0])[0]
    for filename in filenames[1:]:
        tsdss_tmp = cache.load (filename)[0]
        #tsdss = [[tsd + tsd_tmp for tsd, tsd_tmp in izip (tsds, tsds_tmp)]
        #    for tsds, tsds_tmp in izip (tsdss, tsdss_tmp)]
        tsdss += tsdss_tmp
    return (tsdss)

def do_trials_ptw (n_trials, t1, t2, analyses,
                   llh_type=LlhType.per_confchan,
                   mu=0.,
                   use_diffsig=False,
                   n_sig_sources=-1,
                   seed=0):

    """
    Wrapper for :ref:`llh.do_trials` for use with a set of AutoAnalysis instances

    :type   n_trials: int
    :param  n_trials: Number of trials to perform

    :type   analyses: list of AutoAnalysis
    :param  analyses: A list of AutoAnalysis instances to perform pseudo-trials for

    :type   t1: list of doubles, or a list of lists of lists of doubles
    :param  t1: Start of multi-time window search windows, defined universally
        (one list), or per-thrower, per-source (list of lists of lists)

    :type   t2: list of doubles, or a list of lists of lists of doubles
    :param  t2: End of multi-time window search windows, defined universally
        (one list), or per-thrower, per-source (list of lists of lists)

    :type   llh_type: LlhType
    :param  llh_type: One of LlhType.overall, LlhType.per_confchan,
        LlhType.per_source, LlhType.max_confchan, LlhType.max_source.

    :type   mu: number
    :param  mu: Fraction by which to scale the signal weights. `mu = 0.` is the
        default, performing background-only trials.

    :type   use_diffsig: bool
    :param  use_diffsig: If True, use diffuse signal throwers.

    :type   n_sig_sources: int
    :param  n_sig_sources: Number of signal sources to select randomly for
        signal injection (default: use all sources)

    :type   seed: int
    :param  seed: The random number generator seed. (seed % 2**15 is
        actually used to avoid integer overflow.)

    :return: A TestStatDist instance.

    This function conducts pseudo experiments using the given set of
    AutoAnalysis instances.  If any entries in throwers are None, they are
    simply ignored (this behavior is useful for, e.g., :ref:`find_sig_thresh`
    if the background is assumed to be negligible).
    """

    bg_throwers = [analysis.bg_thrower for analysis in analyses]

    if mu == 0.:
        sig_throwers = []
    elif use_diffsig:
        sig_throwers = list (chain.from_iterable ([
            analysis.diffsig_throwers for analysis in analyses]))
        sig_throwers = [mu*thrower for thrower in sig_throwers]
    else:
        sig_throwers = list (chain.from_iterable ([
            analysis.pssig_throwers for analysis in analyses]))
        sig_throwers = [mu*thrower for thrower in sig_throwers]

    return (llh.do_trials_ptw (n_trials, t1, t2, bg_throwers,
                               sig_throwers=sig_throwers,
                               llh_type=llh_type,
                               n_sig_sources=n_sig_sources,
                               seed=seed))

def tsds_from_files_ptw (filenames):
    tsds = [cache.load (filenames[0])]
    for filename in filenames[1:]:
        tsds_tmp = [cache.load (filename)]
        assert (len (tsds) == len (tsds_tmp))
        tsds = [tsd + tsd_tmp for tsd, tsd_tmp in izip (tsds, tsds_tmp)]
    return tsds

def do_trials_p_of_ps (n_trials, tsdss,
                       analyses=None,
                       mu=0.,
                       use_diffsig=False,
                       n_sig_sources=-1,
                       seed=0):

    """
    Wrapper for :ref:`llh.do_trials_p_of_ps` for use with a set of AutoAnalysis instances

    :type   n_trials: int
    :param  n_trials: Number of trials to perform

    :type   tsdss: list of lists of llh.TestStatSF
    :param  tsdss: one or more lists of llh.TestStatSF instances (one list per
        conf/channel, in same order as bg_throwers list, the output of
        :ref:`do_trials_pb`)

    :type   analyses: list of AutoAnalysis or None
    :param  analyses: A list of AutoAnalysis instances to perform pseudo-trials for;
        if None, do trials just with the test statistic distributions.

    :type   mu: number
    :param  mu: Fraction by which to scale the signal weights. `mu = 0.` is the
        default, performing background-only trials.

    :type   use_diffsig: bool
    :param  use_diffsig: If True, use diffuse signal throwers.

    :type   n_sig_sources: int
    :param  n_sig_sources: Number of signal sources to select randomly for
        signal injection (default: use all sources)

    :type   seed: int
    :param  seed: The random number generator seed. (seed % 2**15 is
        actually used to avoid integer overflow.)

    :return: A TestStatDist instance.

    This function conducts pseudo experiments using the given set of
    AutoAnalysis instances.  If any entries in throwers are None, they are
    simply ignored (this behavior is useful for, e.g., :ref:`find_sig_thresh`
    if the background is assumed to be negligible).
    """

    if analyses:
        bg_throwers = [analysis.bg_thrower for analysis in analyses]

        if mu == 0.:
            sig_throwers = []
        elif use_diffsig:
            sig_throwers = list (chain.from_iterable ([
                analysis.diffsig_throwers for analysis in analyses]))
            sig_throwers = [mu*thrower for thrower in sig_throwers]
        else:
            sig_throwers = list (chain.from_iterable ([
                analysis.pssig_throwers for analysis in analyses]))
            sig_throwers = [mu*thrower for thrower in sig_throwers]

    else:
        bg_throwers = None
        sig_throwers = None

    return (llh.do_trials_p_of_ps (n_trials, tsdss,
                                   bg_throwers=bg_throwers,
                                   sig_throwers=sig_throwers,
                                   n_sig_sources=n_sig_sources,
                                   seed=seed))

def do_trials_p_of_ps_tw (n_trials, tsds, t1, t2, analyses,
                          llh_type=LlhType.per_confchan,
                          mu=0.,
                          use_diffsig=False,
                          n_sig_sources=-1,
                          seed=0):

    """
    Wrapper for :ref:`llh.do_trials_p_of_ps` for use with a set of AutoAnalysis instances

    :type   n_trials: int
    :param  n_trials: Number of trials to perform

    :type   tsds: list of llh.TestStatDist
    :param  tsds: one or more lists of llh.TestStatDist instances (one list per
        time window, the output of :ref:`do_trials_ptw`)

    :type   t1: list of doubles, or a list of lists of lists of doubles
    :param  t1: Start of multi-time window search windows, defined universally
        (one list), or per-thrower, per-source (list of lists of lists)

    :type   t2: list of doubles, or a list of lists of lists of doubles
    :param  t2: End of multi-time window search windows, defined universally
        (one list), or per-thrower, per-source (list of lists of lists)

    :type   analyses: list of AutoAnalysis or None
    :param  analyses: A list of AutoAnalysis instances to perform pseudo-trials for;
        if None, do trials just with the test statistic distributions.

    :type   llh_type: LlhType
    :param  llh_type: One of LlhType.overall, LlhType.per_confchan,
        LlhType.per_source, LlhType.max_confchan, LlhType.max_source.

    :type   mu: number
    :param  mu: Fraction by which to scale the signal weights. `mu = 0.` is the
        default, performing background-only trials.

    :type   use_diffsig: bool
    :param  use_diffsig: If True, use diffuse signal throwers.

    :type   n_sig_sources: int
    :param  n_sig_sources: Number of signal sources to select randomly for
        signal injection (default: use all sources)

    :type   seed: int
    :param  seed: The random number generator seed. (seed % 2**15 is
        actually used to avoid integer overflow.)

    :return: A TestStatDist instance.

    This function conducts pseudo experiments using the given set of
    AutoAnalysis instances.  If any entries in throwers are None, they are
    simply ignored (this behavior is useful for, e.g., :ref:`find_sig_thresh`
    if the background is assumed to be negligible).
    """

    bg_throwers = [analysis.bg_thrower for analysis in analyses]

    if mu == 0.:
        sig_throwers = []
    elif use_diffsig:
        sig_throwers = list (chain.from_iterable ([
            analysis.diffsig_throwers for analysis in analyses]))
        sig_throwers = [mu*thrower for thrower in sig_throwers]
    else:
        sig_throwers = list (chain.from_iterable ([
            analysis.pssig_throwers for analysis in analyses]))
        sig_throwers = [mu*thrower for thrower in sig_throwers]

    return (llh.do_trials_p_of_ps_tw (n_trials, tsds, t1, t2, bg_throwers,
                                      sig_throwers=sig_throwers,
                                      llh_type=llh_type,
                                      n_sig_sources=n_sig_sources,
                                      seed=seed))

def find_sig_thresh (thresh, beta, n_trials, analyses,
                     t1=[], t2=[],
                     mu_top=1., mu_bottom=0.,
                     llh_type=LlhType.per_confchan,
                     use_diffsig=False,
                     n_sig_sources=-1,
                     log=False,
                     full_output=False,
                     seed=0):
    """Wrapper for :ref:`llh.find_sig_thresh` for use with a set of
    AutoAnalysis instances

    :type   thresh: number
    :param  thresh: The test statistic threshold.

    :type   beta: number
    :param  beta: Fraction of trials that should pass threshold.

    :type   n_trials: number
    :param  n_trials: The number of trials per normalization iteration.

    :type   analyses: list of AutoAnalysis
    :param  analyses: A list of AutoAnalysis instances to perform pseudo-trials for

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

    :type   use_diffsig: bool
    :param  use_diffsig: If True, use diffuse signal throwers.

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

    bg_throwers = [analysis.bg_thrower for analysis in analyses]

    if use_diffsig:
        sig_throwers = list (chain.from_iterable ([
            analysis.diffsig_throwers for analysis in analyses]))
    else:
        sig_throwers = list (chain.from_iterable ([
            analysis.pssig_throwers for analysis in analyses]))

    return (llh.find_sig_thresh (
        thresh, beta, n_trials, bg_throwers, sig_throwers,
        t1=t1, t2=t2,
        mu_top=mu_top, mu_bottom=mu_bottom,
        llh_type=llh_type,
        n_sig_sources=n_sig_sources,
        log=log,
        full_output=full_output,
        seed=seed))

def find_sig_thresh_p_of_ps (thresh, beta, n_trials, tsdss, analyses,
                             mu_top=1., mu_bottom=0.,
                             llh_type=LlhType.per_confchan,
                             use_diffsig=False,
                             n_sig_sources=-1,
                             log=False,
                             full_output=False,
                             seed=0):
    """Wrapper for :ref:`llh.find_sig_thresh_p_of_ps` for use with a set of
    AutoAnalysis instances

    :type   thresh: number
    :param  thresh: The test statistic threshold.

    :type   beta: number
    :param  beta: Fraction of trials that should pass threshold.

    :type   n_trials: number
    :param  n_trials: The number of trials per normalization iteration.

    :type   tsdss: list of lists of llh.TestStatSF
    :param  tsdss: one or more lists of llh.TestStatSF instances (one list per
        conf/channel, in same order as bg_throwers list, the output of
        :ref:`do_trials_pb`)

    :type   analyses: list of AutoAnalysis
    :param  analyses: A list of AutoAnalysis instances to perform pseudo-trials for

    :type   mu_top: number
    :param  mu_top: Fraction of sig_thrower normalization to treat as initial
        upper bound for finding mu.

    :type   mu_bottom: number
    :param  mu_bottom: Fraction of sig_thrower normalization to treat as
        initial lower bound for finding mu.

    :type   llh_type: LlhType
    :param  llh_type: One of LlhType.overall, LlhType.per_confchan,
        LlhType.per_source, LlhType.max_confchan, LlhType.max_source.

    :type   use_diffsig: bool
    :param  use_diffsig: If True, use diffuse signal throwers.

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

    bg_throwers = [analysis.bg_thrower for analysis in analyses]

    if use_diffsig:
        sig_throwers = list (chain.from_iterable ([
            analysis.diffsig_throwers for analysis in analyses]))
    else:
        sig_throwers = list (chain.from_iterable ([
            analysis.pssig_throwers for analysis in analyses]))

    return (llh.find_sig_thresh_p_of_ps (
        thresh, beta, n_trials, tsdss, bg_throwers, sig_throwers,
        mu_top=mu_top, mu_bottom=mu_bottom,
        llh_type=llh_type,
        n_sig_sources=n_sig_sources,
        log=log,
        full_output=full_output,
        seed=seed))

def find_sig_thresh_p_of_ps_tw (thresh, beta, n_trials, tsds, t1, t2, analyses,
                             mu_top=1., mu_bottom=0.,
                             llh_type=LlhType.per_confchan,
                             use_diffsig=False,
                             n_sig_sources=-1,
                             log=False,
                             full_output=False,
                             seed=0):
    """Wrapper for :ref:`llh.find_sig_thresh_p_of_ps` for use with a set of
    AutoAnalysis instances

    :type   thresh: number
    :param  thresh: The test statistic threshold.

    :type   beta: number
    :param  beta: Fraction of trials that should pass threshold.

    :type   n_trials: number
    :param  n_trials: The number of trials per normalization iteration.

    :type   tsds: list of llh.TestStatDist
    :param  tsds: one or more lists of llh.TestStatDist instances (one list per
        time window, the output of :ref:`do_trials_ptw`)

    :type   t1: list of double
    :param  t1: Start of multi-time window search windows

    :type   t2: list of double
    :param  t2: End of multi-time window search windows

    :type   analyses: list of AutoAnalysis
    :param  analyses: A list of AutoAnalysis instances to perform pseudo-trials for

    :type   mu_top: number
    :param  mu_top: Fraction of sig_thrower normalization to treat as initial
        upper bound for finding mu.

    :type   mu_bottom: number
    :param  mu_bottom: Fraction of sig_thrower normalization to treat as
        initial lower bound for finding mu.

    :type   llh_type: LlhType
    :param  llh_type: One of LlhType.overall, LlhType.per_confchan,
        LlhType.per_source, LlhType.max_confchan, LlhType.max_source.

    :type   use_diffsig: bool
    :param  use_diffsig: If True, use diffuse signal throwers.

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

    bg_throwers = [analysis.bg_thrower for analysis in analyses]

    if use_diffsig:
        sig_throwers = list (chain.from_iterable ([
            analysis.diffsig_throwers for analysis in analyses]))
    else:
        sig_throwers = list (chain.from_iterable ([
            analysis.pssig_throwers for analysis in analyses]))

    return (llh.find_sig_thresh_p_of_ps_tw (
        thresh, beta, n_trials, tsds, t1, t2, bg_throwers, sig_throwers,
        mu_top=mu_top, mu_bottom=mu_bottom,
        llh_type=llh_type,
        n_sig_sources=n_sig_sources,
        log=log,
        full_output=full_output,
        seed=seed))

def observed_test_statistic (ontime_data, analyses,
                             llh_type=LlhType.per_confchan,
                             full_output=False):
    """
    Wrapper of :ref`llh.observed_test_statistic` for easy unblinding of a
    multiple configuration/channels' on-time data.

    :type   ontime_data: list of Container
    :param  ontime_data: list of arbitrary container class containing 'zenith',
        'azimuth', 'sigma', and 'energy' arrays for on-time data events
        (accessible by either ontime_data['key'] or ontime_data.key.
        Optional parameters: 'livetime' (analysis livetime), 't' (event
        times), 'run' (run containing a given event) for each analysis

    :type   analyses: list of AutoAnalysis
    :param  analyses: A list of AutoAnalysis instances to analyze

    :type   llh_type: LlhType
    :param  llh_type: One of LlhType.overall, LlhType.per_confchan,
        LlhType.per_source, LlhType.max_confchan, LlhType.max_source.

    :type   full_output: bool
    :param  full_output: If false, return T alone. If true, return a dict
        containing T, an array of events used, and a corresponding array of
        bursts the events were used for.
    """

    # Setup on-time data events
    ontime_events = [Events (
        _get (data, 'zenith'), _get (data, 'azimuth'),
        _get (data, 'sigma'), _get (data, 'energy'),
        livetime=_get (data, 'livetime'),
        t=_get (data, 't'),
        run=_get (data, 'run')) for data in ontime_data]
    sources = [analysis.bg_sources for analysis in analyses]
    psbs = [analysis.pdf_space_bg for analysis in analyses]
    pres = [analysis.pdf_ratio_energy for analysis in analyses]
    source_n_bs = [analysis.bg_thrower.source_n_b.tolist ()
                   for analysis in analyses]

    return llh.observed_test_statistic (
        ontime_events, sources, psbs, pres, source_n_bs, analyses[0].config,
        llh_type=llh_type,
        full_output=full_output)

def observed_time_windows_max_tw (ontime_data, analyses, t1, t2,
                                   llh_type=LlhType.per_confchan,
                                   full_output=False):
    """
    Wrapper of :ref`llh.observed_test_statistic` for easy unblinding of a
    multiple configuration/channels' on-time data.

    :type   ontime_data: list of Container
    :param  ontime_data: list of arbitrary container class containing 'zenith',
        'azimuth', 'sigma', and 'energy' arrays for on-time data events
        (accessible by either ontime_data['key'] or ontime_data.key.
        Optional parameters: 'livetime' (analysis livetime), 't' (event
        times), 'run' (run containing a given event) for each analysis

    :type   analyses: list of AutoAnalysis
    :param  analyses: A list of AutoAnalysis instances to analyze

    :type   t1: list of doubles
    :param  t1: one or more time window start times relative to source start times

    :type   t2: list of doubles
    :param  t2: one or more time window end times relative to source start times

    :type   llh_type: LlhType
    :param  llh_type: One of LlhType.overall, LlhType.per_confchan,
        LlhType.per_source, LlhType.max_confchan, LlhType.max_source.

    :type   full_output: bool
    :param  full_output: If false, return T alone. If true, return a dict
        containing T, an array of events used, and a corresponding array of
        bursts the events were used for.
    """

    # Setup on-time data events
    ontime_events = [Events (
        _get (data, 'zenith'), _get (data, 'azimuth'),
        _get (data, 'sigma'), _get (data, 'energy'),
        livetime=_get (data, 'livetime'),
        t=_get (data, 't'),
        run=_get (data, 'run')) for data in ontime_data]
    sources = [analysis.bg_sources for analysis in analyses]
    psbs = [analysis.pdf_space_bg for analysis in analyses]
    pres = [analysis.pdf_ratio_energy for analysis in analyses]
    source_rates = [analysis.bg_thrower.source_n_b / analysis.bg_thrower.source_duration
                    for analysis in analyses]

    return llh.observed_time_windows_max_tw (
        ontime_events, sources, t1, t2, psbs, pres, source_rates,
        analyses[0].config,
        llh_type=llh_type,
        full_output=full_output)

def observed_time_windows_p_tw (ontime_data, analyses, tsds, t1, t2,
                               llh_type=LlhType.per_confchan,
                               full_output=False):
    """
    Wrapper of :ref`llh.observed_test_statistic` for easy unblinding of a
    multiple configuration/channels' on-time data.

    :type   ontime_data: list of Container
    :param  ontime_data: list of arbitrary container class containing 'zenith',
        'azimuth', 'sigma', and 'energy' arrays for on-time data events
        (accessible by either ontime_data['key'] or ontime_data.key.
        Optional parameters: 'livetime' (analysis livetime), 't' (event
        times), 'run' (run containing a given event) for each analysis

    :type   analyses: list of AutoAnalysis
    :param  analyses: A list of AutoAnalysis instances to analyze

    :type   tsds: list of TestStatDist
    :param  tsds: one or more TestStatDist instances (one per time window,
        in same order as time window lists)

    :type   t1: list of doubles
    :param  t1: one or more time window start times relative to source start times

    :type   t2: list of doubles
    :param  t2: one or more time window end times relative to source start times

    :type   llh_type: LlhType
    :param  llh_type: One of LlhType.overall, LlhType.per_confchan,
        LlhType.per_source, LlhType.max_confchan, LlhType.max_source.

    :type   full_output: bool
    :param  full_output: If false, return T alone. If true, return a dict
        containing T, an array of events used, and a corresponding array of
        bursts the events were used for.
    """

    # Setup on-time data events
    ontime_events = [Events (
        _get (data, 'zenith'), _get (data, 'azimuth'),
        _get (data, 'sigma'), _get (data, 'energy'),
        livetime=_get (data, 'livetime'),
        t=_get (data, 't'),
        run=_get (data, 'run')) for data in ontime_data]
    sources = [analysis.bg_sources for analysis in analyses]
    psbs = [analysis.pdf_space_bg for analysis in analyses]
    pres = [analysis.pdf_ratio_energy for analysis in analyses]
    source_rates = [analysis.bg_thrower.source_n_b / analysis.bg_thrower.source_duration
                    for analysis in analyses]

    return llh.observed_time_windows_p_tw (
        ontime_events, sources, tsds, t1, t2, psbs, pres, source_rates,
        analyses[0].config,
        llh_type=llh_type,
        full_output=full_output)

def load_analysis (path, confchan):
    a = cache.load ('{0}/{1}.analysis'.format (path, confchan))
    a.set_save_path (path)
    return a


class AutoAnalysis (object):

    """ Automatic analysis constructor. """

    def __init__ (self, confchan, bg_events, bg_sources, diffsim,
                  weights=None,
                  sim_downsample_frac=None,
                  sim_source_smear=False,
                  analysis_zen_range=None,
                  pssim_frac_of_sphere=0.01,
                  grbs_runs=None,
                  seasonal_fit='per_run',
                  seed=None):
        """
        Construct a AutoAnalysis instance.

        :type   confchan: str
        :param  confchan: The name of this configuration+channel (e.g.
            "2008_northern_tracks", "2011_cascades")

        :type   bg_events: Container
        :param  bg_events: Arbitrary container class containing 'zenith',
            'azimuth', 'sigma', and 'energy' arrays, and a `livetime` value for
            background events (accessible by either bg_events['key'] or
            bg_events.key.  Optional parameters: 't' (event times), 'run' (run
            containing a given event)

        :type   bg_sources: Container
        :param  bg_sources: Arbitrary container class containing 'zenith',
            'azimuth', 'sigma', and 'duration' arrays for background sources
            (accessible by either bg_sources['key'] or bg_sources.key.
            Optional parameters:  't' (source start times), 'gbm_pos' (bools of
            whether source localized by GBM, 'run' (run that contains the
            GRB))

        :type   diffsim: Container or list of Containers
        :param  diffsim: Arbitrary container class containing 'zenith',
            'azimuth', 'sigma', 'energy', 'true_zenith', 'true_azimuth',
            'true_energy', 'primary_type', and 'oneweight_by_ngen' arrays for
            simulation events (accessible by either diffsim['key'] or diffsim.key.

        :type   weights: dict or list of dicts
        :param  weights: A dict or list of dicts of diffuse flux weighting
            arrays, one for each diffsim container.

        :type   sim_downsample_frac: double or list of doubles
        :param  sim_downsample_frac: A double or list of doubles giving the
            fraction of each the diffuse simulation datasets to use in the analysis.
            This is applied when creating the simulation events and sources.

        :type   sim_source_smear: bool
        :param  sim_source_smear: If True, smear the true simulation source
            locations according to background source given errors

        :type   analysis_zen_range: list, or tuple, or array
        :param  analysis_zen_range: Range of zenith (in radians) which data is
            available in.  If not set, assumed to be [0, pi].

        :type   pssim_frac_of_sphere: double
        :param  pssim_frac_of_sphere: Fraction of sphere area diffuse events are
            selected around a set of background sources when setting pseudo-point
            source simulation

        :type   smear: bool
        :param  smear: If True, smear the true simulation source locations
            according to background source given errors.

        :type   seed: int
        :param  seed: seed for random source assignment in diffuse simulation,
            and simulation source smearing
        """

        self._confchan = confchan
        self._source_path = None
        self._pdf_space_bg = None
        self._pdf_ratio_energy = None
        self._bg_thrower = None
        self._diffsig_throwers = None
        self._pssig_throwers = None
        self._weight_names = {}
        self._weight_loaded = {}

        # if zenith range not set, assume all-sky
        if analysis_zen_range is None:
            self._analysis_zen_range = np.array ([0, np.pi])
        else:
            self._analysis_zen_range = np.array (sorted (analysis_zen_range))

        self._analysis_omega = 2. * np.pi * (
            np.cos (self._analysis_zen_range.min ()) -
            np.cos (self._analysis_zen_range.max()))

        # Setup background
        self._bg_events = Events (
            _get (bg_events, 'zenith', cast=np.array),
            _get (bg_events, 'azimuth', cast=np.array),
            _get (bg_events, 'sigma', cast=np.array),
            _get (bg_events, 'energy', cast=np.array),
            livetime=_get (bg_events, 'livetime'),
            livetime_per_run=_get(bg_events, 'livetime_per_run'),
            t=_get (bg_events, 't', cast=np.array),
            run=_get (bg_events, 'run', cast=np.array)
        )
        self._bg_sources = Sources (
            _get (bg_sources, 'zenith', cast=np.array),
            _get (bg_sources, 'azimuth', cast=np.array),
            _get (bg_sources, 'sigma', cast=np.array),
            _get (bg_sources, 'duration', cast=np.array),
            t=_get (bg_sources, 't', cast=np.array),
           # _get (bg_sources, 't_100', cast=np.array),
           # t=_get (bg_sources, 't_start', cast=np.array),
            gbm_pos=_get (bg_sources, 'gbm_pos', cast=np.array)
        )

        # If background event time and run information is available, setup
        # background seasonal variation
        if self.bg_events.t is None or self.bg_events.run is None:
            self._bg_rate_vs_time = None
            
        else:
            if grbs_runs is None:
                self._bg_rate_vs_time = SeasonalVariation (
                    self.bg_events.run, self.bg_events.livetime_per_run, self.bg_events.t,
                    #grb_runs=_get (bg_sources, 'run'),
                    duration_min=3600.,
                    width_n_sigmas=2.,
                    averaged_time=seasonal_fit,
                )
            else:
                self._bg_rate_vs_time = SeasonalVariation (
                    self.bg_events.run, self.bg_events.livetime_per_run, self.bg_events.t,
                    grb_runs=grbs_runs,
                    duration_min=3600.,
                    width_n_sigmas=2.,
                    averaged_time=seasonal_fit,
                )

        # Setup signal simulation
        self.set_diffsim (diffsim,
                          weights=weights,
                          downsample_frac=sim_downsample_frac,
                          smear=sim_source_smear,
                          seed=seed)
        self.set_pssim (diffsim,
                        weights=weights,
                        downsample_frac=sim_downsample_frac,
                        frac_of_sphere=pssim_frac_of_sphere,
                        smear=sim_source_smear,
                        seed=seed)


    def set_diffsim (self, diffsim,
                     weights=None,
                     downsample_frac=None,
                     smear=False,
                     seed=None):
        """
        Given background sources set in self.bg_sources, assign random source
        characteristics to diffsim events to create diffuse simulation
        pseudo-sources (one event is one source).

        :type   diffsim: Container or list of Containers
        :param  diffsim: Arbitrary container class containing 'zenith',
            'azimuth', 'sigma', 'energy', 'true_zenith', 'true_azimuth',
            'true_energy', 'primary_type', and 'oneweight_by_ngen' arrays for
            simulation events (accessible by either diffsim['key'] or diffsim.key.

        :type   weights: dict or list of dicts
        :param  weights: A dict or list of dicts of diffuse flux weighting
            arrays, one for each diffsim container.

        :type   downsample_frac: double or list of doubles
        :param  downsample_frac: A double or list of doubles giving the
            fraction of each the diffuse simulation datasets to use in the analysis.
            This is applied when creating the simulation events and sources.

        :type   smear: bool
        :param  smear: If True, smear the true simulation source locations
            according to background source given errors.

        :type   seed: int
        :param  seed: seed for random source assignment and source location
            smearing
        """

        list_diffsim = util._get_list (diffsim)

        if weights is None:
            list_weights = len (list_diffsim) * [{}]
        else:
            list_weights = util._get_list (weights)
        if downsample_frac is None:
            list_downsample_frac = len (list_diffsim) * [1]
        else:
            list_downsample_frac = util._get_list (downsample_frac)

        assert (len (list_diffsim) == len (list_weights))

        self._diffsim_sources = []
        self._diffsim_events = []
        self._diffsim_true_energy = []
        self._diffsim_oneweight_by_ngen_by_omega = []
        self._diffsim_primary_type = []

        for i, (sim, ws, ds_frac) in enumerate (izip (
                list_diffsim, list_weights, list_downsample_frac)):
            n_sim = len (_get (sim, 'zenith'))
            if ds_frac is not 1:
                np.random.seed (seed)
                n_ds = int (ds_frac * n_sim)
                ds_idx = np.arange (n_sim)
                np.random.shuffle (ds_idx)
                ds_idx = ds_idx[:n_ds]
                ds_idx.sort ()
            else:
                ds_idx = np.arange (n_sim)
                n_ds = n_sim
            rand_sources = util.randsamp (
                zip (self.bg_sources.sigma,
                     self.bg_sources.duration,
                     self.bg_sources.gbm_pos,
                     self.bg_sources.source_index),
                n_sim,
                seed=seed)
            
            rand_sigma = np.array (rand_sources[:, 0])
            rand_duration = np.array (rand_sources[:, 1])
            rand_gbm_pos = np.array (rand_sources[:, 2], dtype=bool)
            rand_source_index = np.array (rand_sources[:, 3], dtype=int)

            # Sort diffuse simulation by randomly assigned source index, for
            # use with per-source analyses
            rand_idx = (rand_source_index.argsort ())[ds_idx]
            rand_sigma = rand_sigma[rand_source_index.argsort ()][ds_idx]
            rand_duration = rand_duration[rand_source_index.argsort ()][ds_idx]
            rand_gbm_pos = rand_gbm_pos[rand_source_index.argsort ()][ds_idx]
            rand_source_index.sort ()
            rand_source_index = rand_source_index[ds_idx]
            # Reorder weighting, and make all weighting a per-source fluence
            w_temp = {}
            for w in ws:
                self._weight_loaded[w] = True
                if w in self._weight_names.keys ():
                    self._weight_names[w][i] = True
                else:
                    self._weight_names[w] = np.zeros (len (list_diffsim), dtype=bool)
                    self._weight_names[w][i] = True
                
                w_temp[w] = ws[w][rand_idx] * len (self.bg_sources) / \
                    (ds_frac * self._analysis_omega)

            self._diffsim_events.append (Events (
                _get (sim, 'zenith', cast=np.array)[rand_idx],
                _get (sim, 'azimuth', cast=np.array)[rand_idx],
                _get (sim, 'sigma', cast=np.array)[rand_idx],
                _get (sim, 'energy', cast=np.array)[rand_idx],
                weights=w_temp))
            self._diffsim_sources.append (Sources (
                _get (sim, 'true_zenith', cast=np.array)[rand_idx],
                _get (sim, 'true_azimuth', cast=np.array)[rand_idx],
                rand_sigma, rand_duration,
                gbm_pos=rand_gbm_pos,
                source_index=rand_source_index,
                smear=smear,
                seed=seed)
            )
            self._diffsim_true_energy.append (_get (sim, 'true_energy', cast=np.array)[rand_idx])
            self._diffsim_oneweight_by_ngen_by_omega.append (
                _get (sim, 'oneweight_by_ngen')[rand_idx] /
                (ds_frac * self._analysis_omega) * len (self.bg_sources))
            self._diffsim_primary_type.append (_get (sim, 'primary_type', cast=np.array)[rand_idx])

    def set_pssim (self, diffsim,
                   weights=None,
                   downsample_frac=None,
                   frac_of_sphere=0.01,
                   smear=False,
                   seed=None):
        """
        Given background sources set in self.bg_sources, assign random source
        characteristics to diffsim events to create simulation pseudo-sources.

        :type   diffsim: Container or list of Containers
        :param  diffsim: Arbitrary container class containing 'zenith',
            'azimuth', 'sigma', 'energy', 'true_zenith', 'true_azimuth',
            'true_energy', 'primary_type', and 'oneweight_by_ngen' arrays for
            simulation events (accessible by either diffsim['key'] or diffsim.key.

        :type   weights: dict or list of dicts
        :param  weights: A dict or list of dicts of diffuse flux weighting
            arrays, one for each diffsim container.

        :type   downsample_frac: double or list of doubles
        :param  downsample_frac: A double or list of doubles giving the
            fraction of each the diffuse simulation datasets to use in the analysis.
            This is applied when creating the simulation events and sources.

        :type   frac_of_sphere: double
        :param  frac_of_sphere: Fraction of sphere area diffuse events are
            selected around a set of background sources

        :type   smear: bool
        :param  smear: If True, smear the true simulation source locations
            according to background source given errors.

        :type   seed: int
        :param  seed: seed for source location smearing
        """

        list_diffsim = util._get_list (diffsim)

        if weights is None:
            list_weights = len (list_diffsim) * [{}]
        else:
            list_weights = util._get_list (weights)

        if downsample_frac is None:
            list_downsample_frac = len (list_diffsim) * [1]
        else:
            list_downsample_frac = util._get_list (downsample_frac)

        assert (len (list_diffsim) == len (list_weights))

        self._pssim_events = []
        self._pssim_sources = []
        self._pssim_true_energy = []
        self._pssim_oneweight_by_ngen_by_omega = []
        self._pssim_primary_type = []
        self._pssim_idxs = []

        for (sim, ws, ds_frac) in izip (
                list_diffsim, list_weights, list_downsample_frac):

            source_idxs = []
            idxs = []
            fos_cors = []
            n_sim = len (_get (sim, 'zenith'))

            if ds_frac is not 1:
                np.random.seed (seed)
                n_ds = int (ds_frac * n_sim)
                ds_idx = np.arange (n_sim)
                np.random.shuffle (ds_idx)
                ds_idx = ds_idx[:n_ds]
                ds_idx.sort ()
            else:
                ds_idx = np.arange (n_sim)
                n_ds = n_sim

            for i, source in enumerate (self.bg_sources):
                idx = (llh.pssim_events (source, sim, frac_of_sphere=frac_of_sphere))[ds_idx]
                fos_cor = llh.corrected_frac_of_sphere (frac_of_sphere, source,
                                                        self._analysis_zen_range)

                if fos_cor <= np.finfo(float).eps:
                    idx = np.zeros (len (idx), dtype=bool)
                    fos_cor = 0.
                    logging.warning ("Source {0}, doesn't overlap with specified "
                                     "zenith range. CHECK YOUR CODE, RESULTS "
                                     "MAY NOT MAKE SENSE! Skipping...".format (i))

                fos_cors.append (fos_cor)
                idxs.append (idx)
                source_idxs.append (source.source_index)

            self._pssim_idxs.append (idxs)
            source_index = np.concatenate (tuple (n * np.ones (np.sum (idx), dtype=int)
                        for (idx, n) in izip (idxs, source_idxs)))

            def get_array (name):
                return np.concatenate (
                    tuple (_get (sim, name, cast=np.array)[ds_idx][idx] for idx in idxs))

            pssim_zenith = get_array ('zenith')
            pssim_azimuth = get_array ('azimuth')
            pssim_sigma = get_array ('sigma')
            pssim_energy = get_array ('energy')
            pssim_true_zenith = get_array ('true_zenith')
            pssim_true_azimuth = get_array ('true_azimuth')
            pssim_true_energy = get_array ('true_energy')
            pssim_source_gbm_pos = self.bg_sources.gbm_pos[source_index]
            pssim_source_sigma = self.bg_sources.sigma[source_index]
            pssim_source_duration = self.bg_sources.duration[source_index]

            w_temp = {}
            for w in ws:
                w_temp[w] = np.concatenate (
                    tuple (ws[w][ds_idx][idx] / (4. * np.pi * fos_cor * ds_frac)
                           for (fos_cor, idx) in izip (fos_cors, idxs)))

            self._pssim_events.append (Events (
                pssim_zenith, pssim_azimuth, pssim_sigma, pssim_energy,
                weights=w_temp)
            )
            self._pssim_sources.append (Sources (
                pssim_true_zenith, pssim_true_azimuth,
                pssim_source_sigma, pssim_source_duration,
                gbm_pos=pssim_source_gbm_pos,
                source_index=source_index,
                smear=smear,
                seed=seed)
            )
            self._pssim_true_energy.append (get_array ('true_energy'))
            self._pssim_primary_type.append (get_array ('primary_type'))
            self._pssim_oneweight_by_ngen_by_omega.append (
                np.concatenate (tuple (
                    _get (sim, 'oneweight_by_ngen')[ds_idx][idx] /
                        (4. * np.pi * fos_cor * ds_frac)
                    for (fos_cor, idx) in izip (fos_cors, idxs))))

    def set_pdf_space_bg (self, bins=15, range=None, azimuth=False):
        """ Given background events set in self.bg_events, deterimine the
        background space pdf, set at self.pdf_space_bg.

        :type   bins: int
        :param  bins: The number of bins in the initial histogram.

        :type   range: 2-tuple
        :param  range: A (min(zenith), max(zenith)) tuple.
        """

        self._pdf_space_bg = PDFSpaceBg (self.bg_events, bins=bins, range=range, azimuth=azimuth)

    def add_weighting (self, weight_name, funcs, ntype_factor=1.):
        """
        Add weighting to simulation.

        :type   weight_name: str
        :param  weight_name: Name of the signal weighting to add (must be new)

        :type   funcs: func or list or dict
        :param  funcs: func or list of funcs (one for each source) or dict of
            functions for each simulation `primary_type` which are spectral
            fluence functions with respect to energy.  The functions for each
            `primary_type` can be either a single function, or a list of functions
            for each background source. The weighting is cached in the
            simulation events (both diffuse and pseudo-point source) with key
            `weight_name`.

        :type   ntype_factor: double
        :param  ntype_factor: Factor to get per-nu type weighting in each set of
            diffsim events (e.g. 2 for per-nu/nubar weighting)
        """

        if weight_name in self._weight_names.keys ():
            logging.warning ("Weight being added already exists. Try a new name ...")
            return
        else:
            self._weight_names[weight_name] = np.ones (len (self._diffsim_events), dtype=bool)
            self._weight_loaded[weight_name] = True

        def set_weighting (events, sources, true_energy,
                           oneweight_by_ngen_by_omega, primary_type):
            if isinstance (funcs, dict):
                weights_tmp = np.zeros (len (events))
                for p_type in funcs:
                    try:
                        p_funcs = funcs[p_type]
                        for (i_source, func) in enumerate (p_funcs):
                            idx = ((primary_type == p_type) *
                                   (sources.source_index == i_source))
                            # Extra weighting factor for different particle type
                            # if desired (e.g. 2. for nu vs. nubar)
                            weights_tmp[idx] = ntype_factor * \
                                oneweight_by_ngen_by_omega[idx] * \
                                func (true_energy[idx])
                    except:
                        p_func = funcs[p_type]
                        idx = (primary_type == p_type)
                        # Extra weighting factor for different particle type
                        # if desired (e.g. 2. for nu vs. nubar)
                        weights_tmp[idx] = ntype_factor * \
                            oneweight_by_ngen_by_omega[idx] * \
                            p_func (true_energy[idx])
                events.weights[weight_name] = weights_tmp
            else:
                try:
                    weights_tmp = np.zeros (len (events))
                    for (i_source, func) in enumerate (funcs):
                        idx = (sources.source_index == i_source)
                        # Extra weighting factor for different particle type
                        # if desired (e.g. 2. for nu vs. nubar)
                        weights_tmp[idx] = oneweight_by_ngen_by_omega[idx] * \
                            func (true_energy[idx])
                    events.weights[weight_name] = weights_tmp
                except:
                    events.weights[weight_name] = oneweight_by_ngen_by_omega * \
                        funcs (true_energy)

        for (events, sources, tes, ows, pts) in izip (
                self._diffsim_events,
                self._diffsim_sources,
                self._diffsim_true_energy,
                self._diffsim_oneweight_by_ngen_by_omega,
                self._diffsim_primary_type):
            set_weighting (events, sources, tes, ows, pts)
        for (events, sources, tes, ows, pts) in izip (
                self._pssim_events,
                self._pssim_sources,
                self._pssim_true_energy,
                self._pssim_oneweight_by_ngen_by_omega,
                self._pssim_primary_type):
            set_weighting (events, sources, tes, ows, pts)

    def set_flat_pdf_ratio_energy (self):
        """ Set the energy PDF ratio to be one for all energies."""
        self._pdf_ratio_energy = FlatPDFRatioEnergy ()

    def set_pdf_ratio_energy (self, sig_weight_name,
                              sig_sample=0,
                              use_bg_events=True,
                              bg_weight_name='',
                              bins=30,
                              range=None,
                              pivot=5,
                              **kwargs):
        """ Given background and diffuse simulation events set at self.bg_events
        and self.diffsim_events, determine the energy pdf ratio, set at
        self.pdf_ratio_energy.

        :type   sig_weight_name: str
        :param  sig_weight_name: Name of the signal weighting to use for pdf ratio.

        :type   sig_sample: int
        :param  sig_sample: Index of the diffsim events in self.diffsim_events
            to use for signal and background

        :type   use_bg_events: bool
        :param  use_bg_events: If true, use self.bg_events in pdf ratio

        :type   bg_weight_name: str
        :param  bg_weight_name: Name of the signal weighting to use for pdf ratio.

        :type   bins: int
        :param  bins: The number of bins in the initial histogram.

        :type   range: 2-tuple
        :param  range: A (min(log10(E)), max(log10(E))) tuple.

        :type   pivot: float
        :param  pivot: If using both data and simulation for the
            background PDF, ``pivot`` is the log10(E) where the background
            PDF transitions from data to simulation.

        Either ``use_bg_events`` must be True, or ``bg_weight_name`` must be
        given.  If ``bg_weight_name`` is given and ``use_bg_events`` is False,
        the signal and background energy PDFs will both come from the simulation
        events; otherwise, the ``data_events`` will be used for the background
        energy PDF. If both are given, then data is used for low energies, and
        simulation is used past ``pivot`` energy.

        Additional kwargs are passed to the spline fit
        (scipy.interpolate.UnivariateSpline).
        """

        if sig_weight_name not in self._weight_names.keys ():
            logging.warning ('{0} weighting does not exist. No Energy PDF Ratio set ...'.format (sig_weight_name))
            return
        elif not self._weight_names[sig_weight_name][sig_sample]:
            logging.warning ('{0} weighting does not exist in signal sample selected. No Energy PDF Ratio set ...'.format (sig_weight_name))
            return
        elif not self._weight_loaded[sig_weight_name]:
            self.load_weighting (sig_weight_name)

        if bg_weight_name:
            if bg_weight_name not in self._weight_names.keys ():
                logging.warning ('{0} weighting does not exist. No Energy PDF Ratio set ...'.format (sig_weight_name))
                return
            elif not self._weight_names[bg_weight_name][sig_sample]:
                logging.warning ('{0} weighting does not exist in bg sample selected. No Energy PDF Ratio set ...'.format (sig_weight_name))
                return
            elif not self._weight_loaded[bg_weight_name]:
                self.load_weighting (bg_weight_name)

        if bg_weight_name and use_bg_events:
            self._pdf_ratio_energy = PDFRatioEnergy (
                self.diffsim_events[sig_sample], sig_weight_name,
                bg_weight_name, self.bg_events,
                range=range,
                bins=bins,
                pivot=pivot,
                **kwargs)
        elif bg_weight_name:
            self._pdf_ratio_energy = PDFRatioEnergy (
                self.diffsim_events[sig_sample], sig_weight_name, bg_weight_name,
                range=range,
                bins=bins,
                **kwargs)
        elif use_bg_events:
            self._pdf_ratio_energy = PDFRatioEnergy (
                self.diffsim_events[sig_sample], sig_weight_name,
                data_events=self.bg_events,
                range=range,
                bins=bins,
                **kwargs)
        else:
            logging.warning (
                "No background in energy pdf ratio. Check your inputs ...")
            return

    def set_config (self, **kwargs):
        """Wrapper for :ref:`llh.Config` with an instance stored at
        self.config"""
        self._config = Config (**kwargs)

    def set_bg_thrower (self, energy_bins=10, zenith_bins=10):

        """
        Construct a PseudoBgThrower at self.bg_thower

        :type   energy_bins: int
        :param  energy_bins: The number of energy bins.

        :type   zenith_bins: int
        :param  zenith_bins: The number of zenith bins in each energy bin.
        """

        self._bg_thrower = PseudoBgThrower (
            self.confchan, self.bg_events, self.bg_sources,
            self.pdf_space_bg, self.pdf_ratio_energy,
            energy_bins=energy_bins,
            zenith_bins=zenith_bins,
            config=self.config,
            rate_vs_time=self.bg_rate_vs_time)

    def set_sig_throwers (self, weight_name, mu=1.):
        """Construct a list of SignalThrower's at self.diffsig_throwers (diffuse
        signal throwers) and self.pssig_throwers (pseudo-point source throwers),
        one for each sim sample given.

        :type   weight_name: str
        :param  weight_name: Name of the fluence function passed

        :type   mu: number
        :param  mu: Fraction by which to scale the signal weights.
        """

        for i in range (len (self.diffsim_events)):
            if not self._weight_names[weight_name][i]:
                logging.warning ('{0} weighting does not exist in all entire sim sample. No signal throwers set ...'.format (weight_name))
                return

        if weight_name not in self._weight_names.keys ():
            logging.warning ('{0} weighting does not exist. No signal throwers set ...'.format (weight_name))
            return
        elif not self._weight_loaded[weight_name]:
            self.load_weighting (weight_name)

        self._diffsig_throwers = [
            SignalThrower (
                self.confchan, diffsim_events, weight_name, diffsim_sources,
                self.pdf_space_bg, self.pdf_ratio_energy,
                mu=mu,
                n_sources=diffsim_sources.source_index[-1] + 1,
                config=self.config)
            for (diffsim_events, diffsim_sources) in izip (
                self.diffsim_events, self.diffsim_sources)]
        self._pssig_throwers = [
            SignalThrower (
                self.confchan, pssim_events, weight_name, pssim_sources,
                self.pdf_space_bg, self.pdf_ratio_energy,
                mu=mu,
                n_sources=pssim_sources.source_index[-1] + 1,
                config=self.config)
            for (pssim_events, pssim_sources) in izip (
                self.pssim_events, self.pssim_sources)]

    def load_weighting (self, weight_name):

        if not self._source_path:
            logging.warning ('Analysis source has not been set. Skipping load ...')
            return
        elif weight_name not in self._weight_names.keys ():
            logging.warning ('Weight key not available. Skipping load ...')
            return
        elif self._weight_loaded[weight_name]:
            logging.warning ('Weight already loaded ...')
            return

        dump_dir = '{0}/{1}'.format (self._source_path, self.confchan)

        for i, events in enumerate (self._diffsim_events):
            if self._weight_names[weight_name][i]:
                w = cache.load ('{0}/{1}_{2}.diffsim_weight'.format (
                    dump_dir, weight_name, i))
                events.weights[weight_name] = w

        for i, events in enumerate (self._pssim_events):
            if self._weight_names[weight_name][i]:
                w = cache.load ('{0}/{1}_{2}.pssim_weight'.format (
                    dump_dir, weight_name, i))
                events.weights[weight_name] = w

        self._weight_loaded[weight_name] = True

    def set_save_path (self, path):
        self._source_path = path

    def save (self, path=None, clear=False):

        if path is not None:
            self.set_save_path (path)

        dump_dir = cache.ensure_dir ('{0}/{1}'.format (
            self._source_path, self.confchan))

        for weight_name in self._weight_names.keys ():
            self.save_weighting (weight_name, clear=clear)

        if clear:
            cache.save (self, '{0}/{1}.analysis'.format (
                self._source_path, self.confchan))
        else:
            tmp_analysis = copy.deepcopy (self)
            tmp_analysis._source_path = None

            for weight_name in tmp_analysis._weight_names.keys ():
                for i, events in enumerate (tmp_analysis._diffsim_events):
                    if tmp_analysis._weight_names[weight_name][i]:
                        w = events.weights.pop (weight_name)
                for i, events in enumerate (tmp_analysis._pssim_events):
                    if tmp_analysis._weight_names[weight_name][i]:
                        w = events.weights.pop (weight_name)
                tmp_analysis._weight_loaded[weight_name] = False

            cache.save (tmp_analysis, '{0}/{1}.analysis'.format (
                path, tmp_analysis.confchan))
            del (tmp_analysis)

    def save_weighting (self, weight_name, clear=True):

        if self._source_path is None:
            logging.warning (
                'Save path not set. Set this first with set_save_path(<path>) ...')
            return
        elif weight_name not in self._weight_names.keys ():
            logging.warning ('Weight key not available. Skipping save ...')
            return
        elif not self._weight_loaded[weight_name]:
            logging.warning ('Weight not loaded. Skipping save ...')
            return

        dump_dir = cache.ensure_dir ('{0}/{1}'.format (
            self._source_path, self.confchan))

        if clear:
            for i, events in enumerate (self._diffsim_events):
                if self._weight_names[weight_name][i]:
                    w = events.weights.pop (weight_name)
                    cache.save (w, '{0}/{1}_{2}.diffsim_weight'.format (
                        dump_dir, weight_name, i))

            for i, events in enumerate (self._pssim_events):
                if self._weight_names[weight_name][i]:
                    w = events.weights.pop (weight_name)
                    cache.save (w, '{0}/{1}_{2}.pssim_weight'.format (
                        dump_dir, weight_name, i))

            self._weight_loaded[weight_name] = False
        else:
            for i, events in enumerate (self._diffsim_events):
                if self._weight_names[weight_name][i]:
                    w = events.weights[weight_name]
                    cache.save (w, '{0}/{1}_{2}.diffsim_weight'.format (
                        dump_dir, weight_name, i))

            for i, events in enumerate (self._pssim_events):
                if self._weight_names[weight_name][i]:
                    w = events.weights[weight_name]
                    cache.save (w, '{0}/{1}_{2}.pssim_weight'.format (
                        dump_dir, weight_name, i))

    def observed_test_statistic (self, ontime_data,
                                 llh_type=LlhType.per_confchan,
                                 full_output=False):
        """
        Wrapper of :ref`llh.observed_test_statistic` for easy unblinding of a
        single configuration/channel's on-time data.

        :type   ontime_data: Container
        :param  ontime_data: Arbitrary container class containing 'zenith',
            'azimuth', 'sigma', and 'energy' arrays for on-time data events
            (accessible by either ontime_data['key'] or ontime_data.key.
            Optional parameters: 'livetime' (analysis livetime), 't' (event
            times), 'run' (run containing a given event)

        :type   llh_type: LlhType
        :param  llh_type: One of LlhType.overall, LlhType.per_confchan,
            LlhType.per_source, LlhType.max_confchan, LlhType.max_source.

        :type   full_output: bool
        :param  full_output: If false, return T alone. If true, return a dict
            containing T, an array of events used, and a corresponding array of
            bursts the events were used for.
        """

        # Setup on-time data events
        ontime_events = Events (
            _get (ontime_data, 'zenith'), _get (ontime_data, 'azimuth'),
            _get (ontime_data, 'sigma'), _get (ontime_data, 'energy'),
            livetime=_get (ontime_data, 'livetime'),
            t=_get (ontime_data, 't'),
            run=_get (ontime_data, 'run')
        )

        return llh.observed_test_statistic (
            ontime_events, self.bg_sources, self.pdf_space_bg,
            self.pdf_ratio_energy, self.bg_thrower.source_n_b, self.config,
            llh_type=llh_type,
            full_output=full_output)

    @property
    def bg_thrower (self):
        """The background thrower."""
        return self._bg_thrower

    @property
    def diffsig_throwers (self):
        """The signal thrower list."""
        return self._diffsig_throwers

    @property
    def pssig_throwers (self):
        """The signal thrower list."""
        return self._pssig_throwers

    @property
    def config (self):
        """The configuration-channel of analysis."""
        return self._config

    @property
    def confchan (self):
        """The configuration-channel of analysis."""
        return self._confchan

    @property
    def pdf_space_bg (self):
        """The background space pdf."""
        return self._pdf_space_bg

    @property
    def pdf_ratio_energy (self):
        """The energy pdf ratio."""
        return self._pdf_ratio_energy

    @property
    def bg_rate_vs_time (self):
        """The background rate vs time fit."""
        return self._bg_rate_vs_time

    @property
    def bg_events (self):
        """The background events."""
        return self._bg_events

    @property
    def bg_sources (self):
        """The background sources."""
        return self._bg_sources

    @property
    def diffsim_events (self):
        """The diffuse simulation events."""
        return self._diffsim_events

    @property
    def diffsim_sources (self):
        """The diffuse simulation sources."""
        return self._diffsim_sources

    @property
    def pssim_events (self):
        """The pseudo-point source simulation events."""
        return self._pssim_events

    @property
    def pssim_sources (self):
        """The pseudo-point source simulation sources."""
        return self._pssim_sources
