# Analysis Scripts

Scripts beginning with `frb_` run background (`bg_`) or signal (`sig_`) trial simulations for the stacking and max-burst analyses.<br>
In the background scripts, events are drawn randomly from a Poisson distribution and assigned zenith and azimuth from the background PDF. Then, the test statistic is calculated, and this process is repeated many times to form a distribution of background observations. <br>
The signal scripts perform a binary search for the sensitivity, which is the injected neutrino flux that causes 90% of trial test statistics to exceed the median of the background distribution. 
