from JSON_Classifier import JSON_Classifier
from Music_parser import Music_parser
from Chroma_postprocessing import compute_CENS_from_chromagram
from Dynamic_Time_Warping import compute_matching_function_dtw
from Dynamic_Time_Warping import compute_cost_matrix
from Dynamic_Time_Warping import mininma_from_matching_function
from Dynamic_Time_Warping import matches_dtw
from Dynamic_Time_Warping import compute_optimal_warping_path_subsequence_dtw
from Dynamic_Time_Warping import compute_optimal_warping_path_subsequence_dtw_21
import Visualization
from matplotlib.patches import ConnectionPatch

import datetime
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


# Importing audio files
music_parser = Music_parser()
ref_recording, sr = music_parser.readMusicFile('assets/Reference_20sek.wav')
test_recording, sr = music_parser.readMusicFile('assets/Reference_30sek.wav')


# Importing the metadata from the JSON-file
meta_data = JSON_Classifier()
meta_data.readJSON('assets/testdata.json')


# Splitting the audio file in segments, according to the metadata
# segment_list = music_parser.splitReferenceRecording(meta_data.segments, sr, ref_recording,)


# Feature Extraction
ref_length = librosa.get_duration(ref_recording)
test_length = librosa.get_duration(test_recording)
frame_length = 9600
hopsize = 4800
window = 'hann'


# Sample properties
## Compute waveform
t_ref = np.arange(ref_recording.shape[0]) / sr
t_test = np.arange(test_recording.shape[0]) / sr

plot_waveform(t_ref, ref_recording)
plot_waveform(t_test, test_recording)


## Signal power in dB
# power_db_ref = music_parser.compute_power_db(ref_recording, sr, win_len_sec= 0.2)
# power_db_test = music_parser.compute_power_db(test_recording, sr, win_len_sec= 0.2)

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
# plt.ylim(40,100)
# plt.tick_params(direction='in')
# plt.tight_layout()

# Creating each chromagram
ref_chromagram = music_parser.compute_one_chromagram(ref_recording, sr, norm= None, hop_length= hopsize, n_fft= frame_length, window = window)
test_chromagram = music_parser.compute_one_chromagram(test_recording, sr, norm= None, hop_length= hopsize, n_fft= frame_length, window = window)

## Normalization
ref_chromagram = music_parser.compute_one_chromagram(ref_recording, sr, norm= 1, hop_length= hopsize, n_fft= frame_length, window = window)
test_chromagram = music_parser.compute_one_chromagram(test_recording, sr, norm= 1, hop_length= hopsize, n_fft= frame_length, window = window)

# plt.figure(figsize=(15, 5))
# librosa.display.specshow(ref_chromagram , x_axis='time', y_axis='chroma', sr= sr, hop_length= hopsize, cmap= 'gray_r', clim= [0, 1])
# plt.title(r'$\ell_1$-normalized Chromagram, Sample:Reference Recording')
# plt.xlabel('Time (seconds)')
# plt.yticks(np.arange(12), 'C C# D D# E F F# G G# A A# B'.split())
# plt.colorbar()
# plt.tight_layout()

# plt.figure(figsize=(15, 5))
# librosa.display.specshow(test_chromagram , x_axis='time', y_axis='chroma', sr= sr, hop_length= hopsize, cmap= 'gray_r', clim= [0, 1])
# plt.title(r'$\ell_1$-normalized Chromagram, Sample:Test Recording')
# plt.xlabel('Time (seconds)')
# plt.yticks(np.arange(12), 'C C# D D# E F F# G G# A A# B'.split())
# plt.colorbar()
# plt.tight_layout()


# Creating each CENS feature
ell = 21
d = 5
CENS_test, fs = compute_CENS_from_chromagram(test_chromagram, sr, ell= ell, d= d)
CENS_ref, fs = compute_CENS_from_chromagram(ref_chromagram, sr, ell= ell, d= d)

# plt.figure(figsize=(15, 5))
# librosa.display.specshow(CENS_ref , x_axis='time', y_axis='chroma', sr= fs, hop_length= hopsize, cmap= 'gray_r', clim= [0, 1] )
# plt.title(r'CENS$^{%d}_{%d}$-feature, Sample:Reference Recording' % (ell, d))
# plt.xlabel('Time (minutes)')
# plt.yticks(np.arange(12), 'C C# D D# E F F# G G# A A# B'.split())
# plt.colorbar()
# plt.tight_layout()

# plt.figure(figsize=(15, 5))
# librosa.display.specshow(CENS_test , x_axis='time', y_axis='chroma', sr= fs, hop_length= hopsize, cmap= 'gray_r', clim= [0, 1] )
# plt.title(r'CENS$^{%d}_{%d}$-feature, Sample:Test Recording' % (ell, d))
# plt.xlabel('Time (minutes)')
# plt.yticks(np.arange(12), 'C C# D D# E F F# G G# A A# B'.split())
# plt.colorbar()
# plt.tight_layout()


# Matching
step_size1 = np.array([[1, 0], [0, 1], [1, 1]])
step_size2 = np.array([[1, 1], [2, 1], [1, 2]])
N, M = CENS_ref.shape[1], CENS_test.shape[1]
D, P = librosa.sequence.dtw(X= CENS_ref, Y= CENS_test, metric= 'euclidean', step_sizes_sigma= step_size1, subseq= True, backtrack= True)
Delta = D[-1, :] / N
pos = mininma_from_matching_function(Delta, rho= N//2, tau=0.2)
matches = matches_dtw(pos, D, stepsize= 1)
# print(Delta, '\n\n', Delta1)
# print('\n', D, '\n\n', D1)
# print('\n', P, '\n\n', P1)
print(pos)
print(matches)
b_ast = D[-1, :].argmin()
# P from librosa.sequence.dtw starts at the end ( b_ast ) and ends with the beginning ( a_ast )
a_ast = P[-1, 1]



P = np.array(P) 
plt.figure(figsize=(90, 30))
plt.imshow(D, cmap= 'gray_r', origin= 'lower', aspect= 'equal')
plt.plot(P[:, 1], P[:, 0], marker='o', color='r')
plt.clim([0, np.max(D)])
plt.colorbar()
plt.title('$D$ with optimal warping path')
plt.xlabel('Sequence Y')
plt.ylabel('Sequence X')

fig, ax = plt.subplots(2, 1, gridspec_kw={'width_ratios': [1], 'height_ratios': [1, 1]}, figsize=(8, 4))
plt.plot(Delta)
plt.ylim(0, 1)
plot_matches(ax[1], matches, Delta, s_marker= '', t_marker= 'o')
plt.tight_layout()
plt.show()


