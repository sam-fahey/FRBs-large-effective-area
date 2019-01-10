import numpy as np
import pickle
import tables

# This script combines all final level simulated data and keeps only
# essential event information to minimize file size for reading/processing

filename = 'NuGen_NuMu_allE_IC86i_L2_1_all_MuEX_filtered.h5'
array = {}
low_files = tables.open_file('NuGen_NuMu_LowE_IC86i_L2_1_all_MuEX_filtered.h5')
med_files = tables.open_file('NuGen_NuMu_MedE_IC86i_L2_1_all_MuEX_filtered.h5')

low_data = low_files.root.I3MCWeightDict.cols
med_data = med_files.root.I3MCWeightDict.cols

array['true_zenith'] = np.append(np.array(low_data.PrimaryNeutrinoZenith), np.array(med_data.PrimaryNeutrinoZenith))
array['true_azimuth'] = np.append(np.array(low_data.PrimaryNeutrinoAzimuth), np.array(med_data.PrimaryNeutrinoAzimuth))
array['true_energy'] = np.append(np.array(low_data.PrimaryNeutrinoEnergy), np.array(med_data.PrimaryNeutrinoEnergy))
array['primary_type'] = np.append(np.array(low_data.PrimaryNeutrinoType), np.array(med_data.PrimaryNeutrinoType))

# "low" energy data processed in 1000 jobs of 50000 simulations
# "med" energy data processed in 1000 jobs of 10000 simulations
array['oneweight_by_ngen'] = np.append(np.array(low_data.OneWeight)/(50000.*1000.), np.array(med_data.OneWeight)/(10000.*1000.))

low_data = low_files.root.MPEFitMuEX.cols
med_data = med_files.root.MPEFitMuEX.cols

array['zenith'] = np.append(np.array(low_data.zenith), np.array(med_data.zenith))
array['azimuth'] = np.append(np.array(low_data.azimuth), np.array(med_data.azimuth))
array['energy'] = np.append(np.array(low_data.energy)+.01, np.array(med_data.energy)+.01)

with open(filename, 'wb') as f: pickle.dump(array, f, pickle.HIGHEST_PROTOCOL)
