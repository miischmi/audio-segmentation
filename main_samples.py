from JSON_Classifier import JSON_Classifier
import Music_parser as music_parser
import postprocessing as post
import Dynamic_Time_Warping as dtw
import visualization as vis
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import preprocessing as pre

print('\033[1m' + 'Debugging Prints' + '\033[0m')

ref_track = 'WAM20_20sek.wav'
test_track = 'WAM21_30sek.wav'

#Importing audio files
ref_recording, sr = music_parser.readMusicFile(f'assets/{ref_track}')
test_recording, sr = music_parser.readMusicFile(f'assets/{test_track}')

#Estimate Tuning
ref_tuning = librosa.estimate_tuning(ref_recording, sr)
test_tuning= librosa.estimate_tuning(test_recording, sr)

#Parameter extraction
ref_length = librosa.get_duration(ref_recording, sr= sr)
test_length = librosa.get_duration(test_recording, sr = sr)
frame_length = 9600
hopsize = int(frame_length/2)
window = 'hann'

# ##Sample properties Visualization
# # Compute waveform Visualization
# t_ref = np.arange(ref_recording.shape[0]) / sr
# t_test = np.arange(test_recording.shape[0]) / sr
# title_r = 'Waveform, Sample: Reference Recording'
# title_t = 'Waveform, Sample: Test Recording'
# vis.plot_waveform(t_ref, ref_recording, title_r)
# vis.plot_waveform(t_test, test_recording, title_t)

# # Signal power in dB Visualization
# power_db_ref = music_parser.compute_power_db(ref_recording, sr, win_len_sec= 0.2)
# power_db_test = music_parser.compute_power_db(test_recording, sr, win_len_sec= 0.2)
# title_r = 'Sound power level over time, Sample:Reference Recording'
# title_t = 'Sound power level over time, Sample:Test Recording'
# vis.plot_powerdb(t_ref, power_db_ref, title_r)
# vis.plot_powerdb(t_test, power_db_test, title_t)

# #STFT vs IIRT Visualization for Reference Recording
# center_freqs, sample_rates = music_parser.mr_frequencies_A0(tuning=0.0)
# D = librosa.iirt(ref_recording, sr=sr, win_length= frame_length, hop_length= hopsize, flayout = 'sos', tuning= ref_tuning, center_freqs= center_freqs, sample_rates = sample_rates)
# C = librosa.stft(ref_recording, n_fft= frame_length, hop_length= hopsize, window= window, center = True, pad_mode = 'constant')
# vis.plot_STFT_vs_IIRT(D, C, sr, hopsize)
# plt.show()
print('Recording successfully imported and parameters extracted')



##Onset detection
center_freqs, sample_rates = music_parser.mr_frequencies_A0(tuning=0.0)
# ref_filtered = librosa.iirt(ref_recording, sr=sr, win_length= frame_length, hop_length= hopsize, flayout = 'sos', tuning= ref_tuning, center_freqs=center_freqs, sample_rates=sample_rates)
test_filtered = librosa.iirt(test_recording, sr=sr, win_length= frame_length, hop_length= hopsize, flayout = 'sos', tuning= test_tuning, center_freqs=center_freqs, sample_rates=sample_rates)

#new Parameters
fs= 22050
fs_frame_length = 4410
fs_hopsize = int(fs_frame_length/2)

#Compute novelty spectrum
# ref_nov, ref_Fs_nov = pre.compute_novelty_spectrum(ref_filtered, Fs=fs, N= fs_frame_length, H= fs_hopsize, gamma=10)
test_nov, test_Fs_nov = pre.compute_novelty_spectrum(test_filtered, Fs=fs, N= fs_frame_length, H= fs_hopsize, gamma=10)

#get Peaks
# ref_peaks, ref_properties = signal.find_peaks(ref_nov, prominence=0.02)
test_peaks, test_properties = signal.find_peaks(test_nov, prominence=0.02)
# ref_T_coef = np.arange(ref_nov.shape[0]) / ref_Fs_nov
test_T_coef = np.arange(test_nov.shape[0]) / test_Fs_nov
# ref_peaks_sec = ref_T_coef[ref_peaks]
test_peaks_sec = test_T_coef[test_peaks]

# #Plot Novelty Function + Peaks
# fig, ax, line = vis.plot_signal(test_nov, test_Fs_nov, color='k', title='Novelty function with detected peaks based on a filter bank + STMSP-transformed signal')
# plt.plot(test_peaks_sec, test_nov[test_peaks], 'ro')

# fig, ax, line = vis.plot_signal(ref_nov, ref_Fs_nov, color='k', title='Novelty function with detected peaks based on a filter bank + STMSP-transformed signal')
# plt.plot(ref_peaks_sec, ref_nov[ref_peaks], 'ro')
# plt.show()

#Use peaks to "cut" the recording
# ref_start_sec= ref_peaks_sec[0]
# if ref_start_sec < 1:
#     ref_start_sec = 0
# elif ref_start_sec > 1:
#     ref_start_sec-=1
# ref_start_feat = int(ref_start_sec*sr)
# ref_end_sec = ref_peaks_sec[len(ref_peaks_sec)-1]
# if ref_end_sec < ref_length-1:
#     ref_end_sec+=1
# ref_end_feat = int(ref_end_sec*sr)
# ref_cut_recording = ref_recording[ref_start_feat:ref_end_feat]

test_start_sec= test_peaks_sec[0]
if test_start_sec < 1:
    test_start_sec = 0
elif test_start_sec > 1:
    test_start_sec-=1
test_start_feat = int(test_start_sec*sr)
test_end_sec = test_peaks_sec[len(test_peaks_sec)-1]
if test_end_sec < test_length-1:
    test_end_sec+=1
test_end_feat = int(test_end_sec*sr)
test_cut_recording = test_recording[test_start_feat:test_end_feat]
print('Onset detection successfully performed')



##Compute Chromagram
#IIRT
ell = 41
d = 10
ref_chromagram = pre.get_chromagram(ref_recording, sr, frame_length, hopsize, tuning= ref_tuning)
test_chromagram = pre.get_chromagram(test_cut_recording, sr, frame_length, hopsize, tuning= test_tuning)

# #STFT
# ell = 21
# d = 5
# ref_chromagram = pre.get_chromagram(ref_cut_recording, sr, frame_length, hopsize, stft=True, window=window)
# test_chromagram = pre.get_chromagram(test_cut_recording, sr, frame_length, hopsize, stft=True, window=window)

# #Plot chromagrams
# title_r = r'$\ell_1$-normalized Chromagram, Sample:Reference Recording'
# title_t = r'$\ell_1$-normalized Chromagram, Sample:Test Recording'
# vis.plot_chromagram(ref_chromagram, sr= sr, title= title_r)
# vis.plot_chromagram(test_chromagram, sr= sr, title= title_t)
print('Chromagram computation successfully performed')



##Key detection
#Create labels and sum chromabands of a 20sek sample of each recording
labels= ['C', 'C#/D\u266D', 'D', 'D#/E\u266D', 'E', 'F', 'F#/G\u266D', 'G', 'G#/A\u266D', 'A', 'A#/B\u266D', 'B']
ref_liste =list(map(sum, ref_chromagram[:201]))
test_liste = list(map(sum, test_chromagram[:201]))
#Get values to be below Zero
operator = 1000
ref_liste[:] = [x / operator for x in ref_liste]
test_liste[:] = [x / operator for x in test_liste]

#Chord recognition
X= np.array(ref_liste)
Y= np.array(test_liste)
ref_chord_sim, ref_chord_max = pre.chord_recognition_template(X)
test_chord_sim, test_chord_max = pre.chord_recognition_template(Y)

#Write returned chord_max in dictionary
ref_key={}
test_key={}
for label in range(0, len(labels)):
    ref_key.update({labels[label]:ref_chord_max[label]})
    test_key.update({labels[label]:test_chord_max[label]})

#Extract Key (value=1) and index of key from dictionary
refmax= str(max(ref_key, key=ref_key.get))
testmax= str(max(test_key, key=test_key.get))
refindex = list(ref_key.keys()).index(refmax)
testindex= list(test_key.keys()).index(testmax)

#Perform shift if necessary
shift = 0
newkey= None
if refmax != testmax:
    if  refindex > testindex:
        #key differences --> cyclic_shift
        shift= refindex-testindex
        test_chromagram = post.cyclic_shift(test_chromagram, shift= shift) 
        test_chord_max = post.cyclic_shift(test_chord_max, shift=shift)

    if refindex < testindex:
        #key differences --> cyclic_shift
        shift= 12-(testindex-refindex)
        test_chromagram = post.cyclic_shift(test_chromagram, shift= shift)
        test_chord_max = post.cyclic_shift(test_chord_max, shift=shift)
    new_key={}
    for label in range(0, len(labels)):
        new_key.update({labels[label]:test_chord_max[label]})
    newkey=str(max(new_key, key=new_key.get))

# #Plot key as barplot
# ref_bar_key= ref_key.keys()
# ref_bar_value= ref_key.values()
# test_bar_value= test_key.values()

# fig = plt.subplots(figsize=(7, 5))
# if refmax != testmax:
#     plt.bar(ref_bar_key, ref_bar_value)
#     plt.bar(ref_bar_key, test_bar_value)
# else:
#     barWidth = 0.25
#     barlen= np.arange(len(ref_bar_key))
#     plt.bar(barlen, ref_bar_value, width=barWidth)
#     plt.bar([x + barWidth for x in barlen], test_bar_value, width=barWidth)
#     plt.xticks([r for r in range(len(ref_bar_key))], labels)
# plt.title("Estimated Key " + ref_track + ', ' + test_track )
# plt.legend([ref_track, test_track])
# plt.tight_layout()
# plt.show()
print('Key detection successfuly performed')



##Creating each CENS feature based on the chromagrams
CENS_ref, fs = post.compute_CENS_from_chromagram(ref_chromagram, Fs=sr, ell= ell, d= d)
CENS_test, fs = post.compute_CENS_from_chromagram(test_chromagram, Fs=sr, ell= ell, d= d)

# #Plot CENS
# title_r = r'CENS$^{%d}_{%d}$-feature, Sample:Reference Recording' % (ell, d)
# title_t = r'CENS$^{%d}_{%d}$-feature, Sample:Test Recording' % (ell, d)
# vis.plot_CENS(CENS_ref, fs= sr, title= title_r)
# vis.plot_CENS(CENS_test, fs= sr, title= title_t)
print('CENS creation successfully performed')



##Console print 
print('-'*100)
print('\033[1m' + 'Reference Recording '+ ref_track + '\033[0m')
print('Tuning deviation: ', ref_tuning)
# print('Onset start:', ref_start_sec, '\nOnset end:', ref_end_sec, 
print('Recording duration in seconds:', ref_length)
print('Estimated key:', refmax)
print('-'*100)

print('\033[1m' + 'Test Recording ' + test_track + '\033[0m')
print('Tuning deviation: ', test_tuning)
print('Onset start:', test_start_sec, '\nOnset end:', test_end_sec, '\nRecording duration in seconds:', len(test_cut_recording)/48000)
print('Estimated key:', testmax, '\nshift in semitones:', shift, '\nAltered Key :', newkey)
print('-'*100)




##Matching
print('\033[1m' + 'DTW-Matching'+ '\033[0m')
step_size1 = np.array([[1, 0], [0, 1], [1, 1]])
step_size2 = np.array([[2, 1], [1, 2], [1, 1]])
N, M = CENS_ref.shape[1], CENS_test.shape[1]

#cost matrix & dtw
C= dtw.cost_matrix_dot(CENS_ref, CENS_test)
D, P = librosa.sequence.dtw(C= C, step_sizes_sigma= step_size2, subseq= True, backtrack= True)

#warping path & matching function
P = P[::-1, :]
Delta = D[-1, :] / N
pos = dtw.mininma_from_matching_function(Delta, rho= N//2, tau= 0.1)
matches = dtw.matches_dtw(pos, D, stepsize= 2)

# Indices (same as matches, but in seconds)
b_ast = round(D[-1, :].argmin()* hopsize / fs + (N / fs) / 2)
a_ast = round(P[0, 1]* hopsize / fs + (N / fs) / 2)
sta_ref= round(ref_recording[0]* hopsize / fs + (N / fs) / 2)
end_ref= round(ref_length)

#Jaccard Index
jaccard= post.relativeOverlap(a_ast, b_ast, sta_ref, end_ref)

#print it aaall
print('DTW distance DTW(CENS_ref, CENS_test):',D[-1, -1])
print('Matches (features):', matches)
dtw.print_formatted_matches(matches, hopsize, fs, N)
print('Jaccard Index: ', jaccard)
fig, ax = plt.subplots(2, 1, gridspec_kw={'width_ratios': [1], 'height_ratios': [1, 1]}, figsize=(8, 4), constrained_layout=True)
vis.plot_accCostMatrix_and_Delta(D, P, Delta, matches, ax,  ref_track, test_track, 1)
plt.show()