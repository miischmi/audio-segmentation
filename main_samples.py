from JSON_Classifier import JSON_Classifier
from Music_parser import Music_parser
import Chroma_postprocessing as chroma
import Dynamic_Time_Warping as dtw
import visualization as vis
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

ref_track = 'WAM20_20sek.wav'
test_track = 'WAM79_2min.wav'

# Importing audio files
music_parser = Music_parser()
ref_recording, sr = music_parser.readMusicFile(f'assets/{ref_track}')
test_recording, sr = music_parser.readMusicFile(f'assets/{test_track}')


# Feature Extraction/Definition
ref_length = librosa.get_duration(ref_recording, sr= sr)
test_length = librosa.get_duration(test_recording, sr = sr)
frame_length = 9600
hopsize = 4800
window = 'hann'

# Sample properties
## Compute waveform
# t_ref = np.arange(ref_recording.shape[0]) / sr
# t_test = np.arange(test_recording.shape[0]) / sr
# title_r = 'Waveform, Sample: Reference Recording'
# title_t = 'Waveform, Sample: Test Recording'
# vis.plot_waveform(t_ref, ref_recording, title_r)
# vis.plot_waveform(t_test, test_recording, title_t)

## Signal power in dB
# power_db_ref = music_parser.compute_power_db(ref_recording, sr, win_len_sec= 0.2)
# power_db_test = music_parser.compute_power_db(test_recording, sr, win_len_sec= 0.2)
# title_r = 'Sound power level over time, Sample:Reference Recording'
# title_t = 'Sound power level over time, Sample:Test Recording'
# vis.plot_powerdb(t_ref, power_db_ref, title_r)
# vis.plot_powerdb(t_test, power_db_test, title_t)





## Filterbank-Method
center_freqs, sample_rates = music_parser.mr_frequencies_A0(tuning=0.0)

##STFT vs IIRT Visualization
# D = librosa.iirt(ref_recording, sr=sr, win_length= frame_length, hop_length= hopsize, flayout = 'sos', center_freqs= center_freqs, sample_rates = sample_rates)
# C = librosa.stft(ref_recording, n_fft= frame_length, hop_length= hopsize, window= window, center = True, pad_mode = 'constant')
# vis.plot_STFT_vs_IIRT(D, C, sr, hopsize)
# plt.show()

# Filter through filterbank
ref_filtered = librosa.iirt(ref_recording, sr=sr, win_length= frame_length, hop_length= hopsize, flayout = 'sos', center_freqs= center_freqs, sample_rates = sample_rates)
ref_filtered = librosa.feature.chroma_cqt(C=ref_filtered, bins_per_octave=12, n_octaves=7, fmin=librosa.midi_to_hz(21), norm=None)
test_filtered = librosa.iirt(test_recording, sr=sr, win_length= frame_length, hop_length= hopsize, flayout = 'sos', center_freqs= center_freqs, sample_rates = sample_rates)
test_filtered = librosa.feature.chroma_cqt(C=test_filtered, bins_per_octave=12, n_octaves=7, fmin=librosa.midi_to_hz(21), norm=None)

#new Parameters
sr= 22050
frame_length = 4410
hopsize = int(frame_length/2)          
window = 'hann'

## key differences --> cyclic_shift
# test_filtered = chroma.cyclic_shift(test_filtered, shift= 1)

# Creating each CENS feature based on the Filterbank-chromagrams
ell = 41
d = 10
CENS_ref, fs = chroma.compute_CENS_from_chromagram(ref_filtered, Fs=sr, ell= ell, d= d)
CENS_test, fs = chroma.compute_CENS_from_chromagram(test_filtered, Fs=sr, ell= ell, d= d)

## Plot CENS
# title_r = r'CENS$^{%d}_{%d}$-feature, Sample:Reference Recording' % (ell, d)
# title_t = r'CENS$^{%d}_{%d}$-feature, Sample:Test Recording' % (ell, d)
# vis.plot_CENS(CENS_ref, fs= 4800, title= title_r)
# vis.plot_CENS(CENS_test, fs= 4800, title= title_t)




##STFT Method
# Chreating each chromagram
# ref_chromagram = music_parser.compute_one_chromagram(ref_recording, sr, norm= None, hop_length= hopsize, n_fft= frame_length, window = window, tuning= 0)
# test_chromagram = music_parser.compute_one_chromagram(test_recording, sr, norm= None, hop_length= hopsize, n_fft= frame_length, window = window, tuning= 0)

## key differences --> cyclic_shift
# test_chromagram = chroma.cyclic_shift(test_chromagram, shift= 1)

## Plot chromagrams
# title_r = r'$\ell_1$-normalized Chromagram, Sample:Reference Recording'
# title_t = r'$\ell_1$-normalized Chromagram, Sample:Test Recording'
# vis.plot_chromagram(ref_chromagram, sr= sr, title= title_r)
# vis.plot_chromagram(test_chromagram, sr= sr, title= title_t)

# Creating each CENS feature based on the STFT-chromagrams
# ell = 21
# d = 5
# CENS_ref, fs = chroma.compute_CENS_from_chromagram(ref_chromagram, Fs=sr, ell= ell, d= d)
# CENS_test, fs = chroma.compute_CENS_from_chromagram(test_chromagram, Fs=sr, ell= ell, d= d)

## Plot CENS
# title_r = r'CENS$^{%d}_{%d}$-feature, Sample:Reference Recording' % (ell, d)
# title_t = r'CENS$^{%d}_{%d}$-feature, Sample:Test Recording' % (ell, d)
# vis.plot_CENS(CENS_ref, fs= 4800, title= title_r)
# vis.plot_CENS(CENS_test, fs= 4800, title= title_t)




# Matching
# step_size1 = np.array([[1, 0], [0, 1], [1, 1]])
step_size2 = np.array([[2, 1], [1, 2], [1, 1]])
N, M = CENS_ref.shape[1], CENS_test.shape[1]
C= dtw.cost_matrix_dot(CENS_ref, CENS_test)
D, P = librosa.sequence.dtw(C= C, step_sizes_sigma= step_size2, subseq= True, backtrack= True)
P = P[::-1, :]
Delta = D[-1, :] / N
pos = dtw.mininma_from_matching_function(Delta, rho= N//2, tau= 0.1)
matches = dtw.matches_dtw(pos, D, stepsize= 2)
print(len(P))
print('DTW distance DTW(CENS_ref, CENS_test):',D[-1, -1])
print(matches)
dtw.print_formatted_matches(matches, hopsize, fs, N)
fig, ax = plt.subplots(2, 1, gridspec_kw={'width_ratios': [1], 'height_ratios': [1, 1]}, figsize=(8, 4), constrained_layout=True)
vis.plot_accCostMatrix_and_Delta(D, P, Delta, matches, ax,  ref_track, test_track, 1)
plt.show()
