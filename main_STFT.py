from JSON_Classifier import JSON_Classifier
import Music_parser as music_parser
import postprocessing as chroma
import Dynamic_Time_Warping as dtw
import visualization as vis
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import datetime

begin_time = datetime.datetime.now()
ref_track = 'WAM20_Segm_1-6.wav'
test_track = 'WAM-21__Track1_Channel1.wav'

# Importing audio files
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

    # Indices (same as matches, but in seconds)
    b_ast = round(D[-1, :].argmin() * hopsize / fs + (N / fs) / 2)
    a_ast = round(P[0, 1] * hopsize / fs + (N / fs) / 2)
    sta_ref= meta_data.segments[i]['start']
    end_ref= meta_data.segments[i]['ende']

    #Jaccard Index
    jaccard= chroma.relativeOverlap(a_ast, b_ast, sta_ref, end_ref)
    
    print('-'*100)
    print('\nSegment ', i+1)
    print('\tJaccard Index: ', jaccard)
    print('\tDTW distance DTW(CENS_ref, CENS_test):',D[-1, -1])
    print('\tMatches (features):', matches)
    dtw.print_formatted_matches(matches, hopsize, fs, N)
    print('\tMatches (seconds): [['+ a_ast + ', '+ b_ast + ']]')
    print('\tQuery (seconds): ['+sta_ref + ', ' + end_ref + ']]')
    fig, ax = plt.subplots(2, 1, gridspec_kw={'width_ratios': [1], 'height_ratios': [1, 1]}, figsize=(8, 4), constrained_layout=True)
    vis.plot_accCostMatrix_and_Delta(D, P, Delta, matches, ax, ref_track, test_track, i)
    plt.savefig('/mnt/smb.hdd.rbd/W/opera4ever/Schmid/STFT_Matching_segment'+ str(i+1)+'.png')
print('-'*100)
print('Skriptdauer: ', datetime.datetime.now() - begin_time)