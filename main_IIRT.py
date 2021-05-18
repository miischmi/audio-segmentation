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
import datetime

begin_time = datetime.datetime.now()
print('\033[1m' + 'Debugging Prints' + '\033[0m')

ref_track = 'WAM20_Segm_1-6.wav'
test_track = 'WAM-21__Track1_Channel1.wav'

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
print('\tRecording successfully imported and parameters extracted')




#Importing the metadata from the JSON-file
meta_data = JSON_Classifier()
meta_data.readJSON('assets/testdata.json')

#Splitting the audio file in segments, according to the metadata
segment_list = music_parser.splitReferenceRecording(meta_data.segments, sr, ref_recording,)
print('\tImporting metadata successfully performed')



#Onset detection
center_freqs, sample_rates = music_parser.mr_frequencies_A0(tuning=0.0)
test_filtered = librosa.iirt(test_recording, sr=sr, win_length= frame_length, hop_length= hopsize, flayout = 'sos', tuning= test_tuning, center_freqs=center_freqs, sample_rates=sample_rates)

#new Parameters
fs= 22050
fs_frame_length = 4410
fs_hopsize = int(fs_frame_length/2)

#Compute novelty spectrum
test_nov, test_Fs_nov = pre.compute_novelty_spectrum(test_filtered, Fs=fs, N= fs_frame_length, H= fs_hopsize, gamma=10)

#get Peaks
test_peaks, test_properties = signal.find_peaks(test_nov, prominence=0.02)
test_T_coef = np.arange(test_nov.shape[0]) / test_Fs_nov
test_peaks_sec = test_T_coef[test_peaks]

#Use peaks to "cut" the recording
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
print('\tOnset detection successfully performed')





# Creating each chromagram
ell = 41
d = 10
ref_chromagram = pre.get_chromagrams(segment_list, sr, frame_length, hopsize, tuning= ref_tuning)
test_chromagram = pre.get_chromagram(test_cut_recording, sr, frame_length, hopsize, tuning= test_tuning)
key_chroma = pre.get_chromagram(test_cut_recording[0:960000], sr, frame_length, hopsize, tuning= test_tuning)
print('\tChromagram computation successfully performed')




##Key detection
#Create labels and sum chromabands of a 20sek sample of each recording
labels= ['C', 'C#/D\u266D', 'D', 'D#/E\u266D', 'E', 'F', 'F#/G\u266D', 'G', 'G#/A\u266D', 'A', 'A#/B\u266D', 'B']
ref_liste =list(map(sum, ref_chromagram[0][0:201]))
test_liste = list(map(sum, key_chroma))
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
print('\tKey detection successfuly performed')




# Creating each CENS feature based on the chromagrams
CENS_refs, fs = post.compute_CENS_from_chromagrams_seg(ref_chromagram, sr, ell= ell, d= d)
CENS_test, fs = post.compute_CENS_from_chromagram(test_chromagram, sr, ell= ell, d= d)
print('\tCENS creation successfully performed')




##Console print 
print('-'*100)
print('\033[1m' + 'Reference Recording '+ ref_track + '\033[0m')
print('\tTuning deviation: ', ref_tuning)
print('\tRecording duration in seconds:', ref_length)
print('\tEstimated key:', refmax)
print('-'*100)

print('\033[1m' + 'Test Recording ' + test_track + '\033[0m')
print('\tTuning deviation: ', test_tuning)
print('\tOnset start:', test_start_sec, '\n\tOnset end:', test_end_sec, '\n\tRecording duration in seconds:', len(test_cut_recording)/48000)
print('\tEstimated key:', testmax, '\n\tshift in semitones:', shift, '\n\tAltered Key :', newkey)
print('-'*100)



#Matching
print('\033[1m' + 'DTW-Matching'+ '\033[0m')
step_size1 = np.array([[1, 0], [0, 1], [1, 1]])
step_size2 = np.array([[2, 1], [1, 2], [1, 1]])
M =  CENS_test.shape[1]

for i in range(len(CENS_refs)):
    N = CENS_refs[i].shape[1]

    #cost matrix & dtw
    C = dtw.cost_matrix_dot(CENS_refs[i], CENS_test)
    D, P = librosa.sequence.dtw(C= C, step_sizes_sigma= step_size2, subseq= True, backtrack= True)
    
    #warping path & matching function
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
    jaccard= post.relativeOverlap(a_ast, b_ast, sta_ref, end_ref)
    
    print('\nSegment ', i+1)
    print('\tJaccard Index: ', jaccard)
    print('\tDTW distance DTW(CENS_ref, CENS_test):',D[-1, -1])
    print('\tMatches (features):', matches)
    dtw.print_formatted_matches(matches, hopsize, fs, N)
    print('\tMatches (seconds): [[', a_ast , ', ', b_ast, ']]')
    print('\tQuery (seconds): [', sta_ref, ', ', end_ref, ']]')
    fig, ax = plt.subplots(2, 1, gridspec_kw={'width_ratios': [1], 'height_ratios': [1, 1]}, figsize=(8, 4), constrained_layout=True)
    vis.plot_accCostMatrix_and_Delta(D, P, Delta, matches, ax, ref_track, test_track, i)
    plt.savefig('/mnt/smb.hdd.rbd/W/opera4ever/Schmid/Matching_segment'+ str(i+1)+'.png')
print('-'*100)
print('Skriptdauer: ', datetime.datetime.now() - begin_time)