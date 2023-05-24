import mne
import numpy as np
import matplotlib.pyplot as plt
from mne import create_info
from mne.io import RawArray

data = np.loadtxt(r'C:\Users\Sarp Akgündüz\PycharmProjects\pythonProject\EEGSignalProccesing\auditory-evoked-potential-eeg-biometric-dataset-1.0.0\Raw_Data\s01_ex01_s01.txt', skiprows=1, delimiter=',', usecols=(1, 2, 3, 4))


ch_names = ['P4', 'Cz', 'F8', 'T7']  # Channel names
sfreq = 200  # Sampling frequency
info = create_info(ch_names, sfreq)

# Create a RawArray object
raw = RawArray(data.T, info)

# Create the Raw object
raw = mne.io.RawArray(data.T, info)

picks = [0, 1, 2, 3]  # Indices of the channels to include

# Apply bandpass filter to the selected channels
raw.filter(1, 40, picks=picks)

# Set the montage
try:
    raw.set_montage('standard_1020')
except ValueError as e:
    if 'channels missing' in str(e):
        print("Some channels are missing from the montage. Ignoring missing channels.")
        raw.set_montage('standard_1020', on_missing='ignore')
    else:
        raise e

# Manually specify the stimulation channel
stim_channel = 'Cz'  # Replace with the appropriate stimulation channel name

# Extract events from the stimulation channel
min_duration = 0.010   # Minimum duration is set to 1 sample duration
events = mne.find_events(raw, stim_channel=stim_channel, min_duration=min_duration,uint_cast=True)

print(f"{len(events)} events found")
print("Event IDs:", sorted(set(events[:, -1])))

# Extract epochs of EEG data
event_id = {'Stimulus/Target': 1, 'Non-Target': 2}
epochs = mne.Epochs(raw, events, event_id, tmin=-0.2, tmax=0.8, baseline=None, preload=True)

# Perform further analysis on the epochs
# e.g., compute event-related potentials (ERPs), spectral analysis, etc.

# Visualize the EEG data and analysis results
epochs.plot(n_channels=10, scalings='auto', picks=picks)

plt.show()