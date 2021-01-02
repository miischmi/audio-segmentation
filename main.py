from JSON_Classifier import JSON_Classifier
from Music_parser import Music_parser
import Chroma_postprocessing as chroma
import Dynamic_Time_Warping as dtw
import visualization as vis
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

ref_track = 'WAM20_Segm_1-6.wav'
test_track = 'WAM-21__Track1_Channel1.wav'

# Importing audio files
music_parser = Music_parser()
ref_recording, sr = music_parser.readMusicFile(f'assets/{ref_track}')
test_recording, sr = music_parser.readMusicFile(f'assets/{test_track}')



# Importing the metadata from the JSON-file
meta_data = JSON_Classifier()
meta_data.readJSON('assets/testdata.json')



# Splitting the audio file in segments, according to the metadata
segment_list = music_parser.splitReferenceRecording(meta_data.segments, sr, ref_recording,)



# Feature Extraction/Definition
ref_length = librosa.get_duration(ref_recording, sr= sr)
test_length = librosa.get_duration(test_recording, sr = sr)
frame_length = 9600
hopsize = 4800
window = 'hann'



# Creating each chromagram
ref_chromagram = music_parser.compute_chromagrams(segment_list, sr, norm= None, hop_length= hopsize, n_fft= frame_length, window = window, tuning= 0)
test_chromagram = music_parser.compute_one_chromagram(test_recording, sr, norm= None, hop_length= hopsize, n_fft= frame_length, window = window, tuning= 0)

## key differences --> cyclic_shift
test_chromagram = chroma.cyclic_shift(test_chromagram, shift= 1)



# Creating each CENS feature based on the chromagrams
ell = 21
d = 5
CENS_refs, fs = chroma.compute_CENS_from_chromagrams_seg(ref_chromagram, sr, ell= ell, d= d)
CENS_test, fs = chroma.compute_CENS_from_chromagram(test_chromagram, sr, ell= ell, d= d)



# Matching
step_size2 = np.array([[2, 1], [1, 2], [1, 1]])
for i in range(len(CENS_refs)):
    N = CENS_refs[i].shape[1]
    C = dtw.cost_matrix_dot(CENS_refs[i], CENS_test)
    D, P = librosa.sequence.dtw(C= C, step_sizes_sigma= step_size2, subseq= True, backtrack= True)
    P = P[::-1, :]
    Delta = D[-1, :] / N
    pos = dtw.mininma_from_matching_function(Delta, rho= N//2, tau= 0.1, num= 3)
    matches = dtw.matches_dtw(pos, D, stepsize= 2)

    # Indices
    b_ast = D[-1, :].argmin()
    a_ast = P[0, 1]

    print(matches)
    dtw.print_formatted_matches(matches, hopsize, fs, N)
    fig, ax = plt.subplots(2, 1, gridspec_kw={'width_ratios': [1], 'height_ratios': [1, 1]}, figsize=(8, 4), constrained_layout=True)
    vis.plot_accCostMatrix_and_Delta(D, P, Delta, matches, ax, ref_track, test_track, i)
    plt.show()