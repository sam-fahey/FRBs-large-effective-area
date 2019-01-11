# Python Software

`cache.py` defines functions for loading and saving test statistic files compatible with IceCube's grbllh format.

`llh.py` defines the functions that:
* evaluate the test statistic or set of test statistics for an experimental observation 
* set analysis spatial boundaries
* set parameters for background and signal simulation
* performs simulations of experimental control and analysis sensitivity
* supports implementation of various analysis types (stacking, max-burst, single-source, multi-channel)

`pdf.py` defines the functions that:
* calculate background probability distribution functions in space and energy
* perform seasonal variation fit to find time-dependent event rate

`util.py` and `fitting.py` define fitting functions and simple tools used for likelihood calculations and PDF fits.

