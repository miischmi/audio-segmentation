from JSON_Classifier import JSON_Classifier
from Music_parser import Music_parser
from Chroma_postprocessing import compute_CENS_from_chromagram
from Chroma_postprocessing import compute_CENS_from_chromagrams_seg

import datetime
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


# Importing audio files
music_parser = Music_parser()
ref_recording, ref_samplerate = music_parser.readMusicFile('assets/Reference_1min.wav')
test_recording, test_samplerate = music_parser.readMusicFile('assets/unbek_2min.wav')


# Importing the metadata from the JSON-file
meta_data = JSON_Classifier()
meta_data.readJSON('assets/testdata.json')


# Splitting the audio file in segments, according to the metadata
# segment_list = music_parser.splitReferenceRecording(meta_data.segments, ref_samplerate, ref_recording,)


# Feature Extraction
ref_length = librosa.get_duration(ref_recording)
test_length = librosa.get_duration(test_recording)
frame_length = 4410
hopsize = 2205
window = 'hann'


# Sample properties
## Compute waveform
# t_ref = np.arange(ref_recording.shape[0]) / ref_samplerate
# t_test = np.arange(test_recording.shape[0]) / test_samplerate

# plt.figure(figsize=(10, 2))
# plt.plot(t_ref, ref_recording, color='gray')
# plt.title('Waveform, Sample:Reference Recording')
# plt.xlabel('Time (seconds)')
# plt.ylabel('Amplitude')
# plt.tick_params(direction='in')
# plt.tight_layout()

# plt.figure(figsize=(10, 2))
# plt.plot(t_test, test_recording, color='gray')
# plt.title('Waveform, Sample:Test Recording')
# plt.xlabel('Time (seconds)')
# plt.ylabel('Amplitude')
# plt.tick_params(direction='in')
# plt.tight_layout()

## Signal power in dB
# power_db_ref = music_parser.compute_power_db(ref_recording, ref_samplerate, win_len_sec= 0.2)
# power_db_test = music_parser.compute_power_db(test_recording, test_samplerate, win_len_sec= 0.2)

# plt.figure(figsize=(10, 2))
# plt.plot(t_ref, power_db_ref, color='red')
# plt.title('Sound power level over time, Sample:Reference Recording')
# plt.xlabel('Time (seconds)')
# plt.ylabel('Power (dB)')
# plt.ylim(40,100)
# plt.tick_params(direction='in')
# plt.tight_layout()

# plt.figure(figsize=(10, 2))
# plt.plot(t_test, power_db_test, color='red')
# plt.title('Sound power level over time, Sample:Test Recording')
# plt.xlabel('Time (seconds)')
# plt.ylabel('Power (dB)')
# plt.ylim(40,100) plt.tick_params(direction='in')
# plt.tight_layout()


# Creating each chromagram
ref_chromagram = music_parser.compute_one_chromagram(ref_recording, ref_samplerate, norm= None, hop_length= hopsize, n_fft= frame_length, window = window)
test_chromagram = music_parser.compute_one_chromagram(test_recording, test_samplerate, norm= None, hop_length= hopsize, n_fft= frame_length, window = window)

## Normalization
ref_chromagram = music_parser.compute_one_chromagram(ref_recording, ref_samplerate, norm= 1, hop_length= hopsize, n_fft= frame_length, window = window)
test_chromagram = music_parser.compute_one_chromagram(test_recording, test_samplerate, norm= 1, hop_length= hopsize, n_fft= frame_length, window = window)

# plt.figure(figsize=(15, 5))
# librosa.display.specshow(test_chromagram , x_axis='time', y_axis='chroma', sr= test_samplerate, hop_length= hopsize, cmap= 'gray_r', clim= [0, 1])
# plt.title(r'$\ell_1$-normalized Chromagram, Sample:Test Recording')
# plt.xlabel('Time (minutes)')
# plt.yticks(np.arange(12), 'C C# D D# E F F# G G# A A# B'.split())
# plt.colorbar()
# plt.tight_layout()


# Creating each CENS feature
ell = 41
d = 10
CENS_test, CENS_test_featurerate = compute_CENS_from_chromagram(test_chromagram, test_samplerate, ell= ell, d= d)
CENS_ref, CENS_ref_featurerate = compute_CENS_from_chromagram(ref_chromagram, ref_samplerate, ell= ell, d= d)

# plt.figure(figsize=(15, 5))
# librosa.display.specshow(CENS_ref , x_axis='time', y_axis='chroma', sr= CENS_ref_featurerate, hop_length= hopsize, cmap= 'gray_r', clim= [0, 1] )
# plt.title(r'CENS$^{%d}_{%d}$-feature, Sample:Reference Recording' % (ell, d))
# plt.xlabel('Time (minutes)')
# plt.yticks(np.arange(12), 'C C# D D# E F F# G G# A A# B'.split())
# plt.colorbar()
# plt.tight_layout()
 

# Matching



