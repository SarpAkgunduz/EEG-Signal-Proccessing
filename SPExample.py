import mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(r'C:\Users\Sarp Akgündüz\PycharmProjects\pythonProject\EEGSignalProccesing\auditory-evoked-potential-eeg-biometric-dataset-1.0.0\Filtered_Data\s01_ex01_s01.csv')

# Create MNE Raw object from the DataFrame
data = df[['P4', 'Cz', 'F8', 'T7']].values  # Assuming your data is in numeric format

# Define the channel names and sampling frequency
channel_names = ['P4', 'Cz', 'F8', 'T7']
sfreq = 200  # Replace with your actual sampling frequency

# Create the info structure required by MNE
info = mne.create_info(channel_names, sfreq, ch_types='eeg')

# Create the Raw object
raw = mne.io.RawArray(data.T, info)

# Preprocessing steps
raw.filter(1, 40)  # Apply a bandpass filter between 1 and 40 Hz

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
epochs.plot(n_channels=10, scalings='auto')

plt.show()