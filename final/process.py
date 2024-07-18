import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import pyedflib as plib
from pyedflib import highlevel

import scipy.io
import scipy.signal

import os

# defining a function to filter the data using a bandpass filter
def bandpass(data: np.ndarray, edges: list[float], sample_rate: float, poles: int = 5):
    sos = scipy.signal.butter(poles, edges, 'bandpass', fs=sample_rate, output='sos')
    filtered_data = scipy.signal.sosfiltfilt(sos, data)
    return filtered_data

def process_edf_file(file_path: str):
    if os.path.getsize(file_path) / 1024 / 1024 > 1000: # File size in MB
        return
    data, data_headers, _ = highlevel.read_edf(file_path)
    # for da in data_headers:
    #     print(da['label'])
    # return

    # Filtering only the channels with labels starting with 'EEG'
    eeg_indices = [i for i, header in enumerate(data_headers) if header['label'].startswith('EEG')]
    eeg_labels = [head['label'] for head in data_headers if head['label'].startswith('EEG')]
    
    data_array = np.array(data)
    
    # Extract the EEG channels using NumPy indexing
    eeg_data = data_array[eeg_indices, :]
    
    # Construct the DataFrame
    df = pd.DataFrame(eeg_data.T, columns=eeg_labels)
    
    print(df.shape)

    # filtering the data for 32Hz and below frequencies
    filt_df = pd.DataFrame(bandpass(df.T, [0.5, 25], 512).T, columns=df.columns)

    # downsampling the data to 64Hz
    original_fs = 512  # Original sampling frequency in Hz
    new_fs = 64  # New sampling frequency in Hz
    new_length = int(len(filt_df) * (new_fs / original_fs))

    # Resample the time series data
    downsampled_df = pd.DataFrame(scipy.signal.resample(filt_df, new_length), columns=filt_df.columns)

    # normalizing the data
    norm_df = (downsampled_df - downsampled_df.mean()) / downsampled_df.std()
    print(norm_df.shape)
    return norm_df


# saving the downsampled data to csv files
# Define EDF file and header
input_path = './final/physionet.org/files/siena-scalp-eeg/1.0.0/'
file_path = 'PN09/PN09-3.edf'
norm = process_edf_file(input_path + file_path)

# exit()

channel_labels = norm.columns.tolist()
n_channels = len(channel_labels)
sfreq = 64  # Example sampling frequency, adjust as needed

def can_write_to_file(file_path):
    if os.path.exists(file_path):
        try:
            with open(file_path, 'a'):
                pass
            return True
        except IOError:
            return False
    else:
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        return os.access(directory, os.W_OK)

output_path = './final/datapoints/' + file_path
# Initialize the header
if can_write_to_file(output_path):
    try:
        with plib.EdfWriter(
        output_path,
        n_channels=n_channels,
        file_type=plib.FILETYPE_EDFPLUS
        ) as header:
            # Define channel information
            channel_info = []
            for label in channel_labels:
                chan_dict = {
                    'label': label,
                    'sample_rate': sfreq,
                    'dimension': 'uV',
                    'physical_min': norm[label].min(),  # Adjusted to actual data range
                    'physical_max': norm[label].max(),  # Adjusted to actual data range
                    'digital_min': -32768,
                    'digital_max': 32767,
                    'transducer': '',
                    'prefilter': ''
                }
                channel_info.append(chan_dict)

            header.setSignalHeaders(channel_info)

            # Write data to EDF file
            header.writeSamples(norm.values.T)
    except Exception as e:
        print(f"Failed to write EDF file: {e}")
else:
    print('File already exists')
