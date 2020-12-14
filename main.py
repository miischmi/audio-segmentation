from JSON_Classifier import JSON_Classifier
from Music_parser import Music_parser
from Chroma_postprocessing import compute_CENS_from_chromagram
from Chroma_postprocessing import compute_CENS_from_chromagrams_seg
import datetime
import librosa
import librosa.display
import matplotlib.pyplot as plt

# t1 = JSON_Classifier()
# t1.readJSON('assets/testdata.json')

music_parser = Music_parser()
musicRef, sample_rate_ref = music_parser.readMusicFile('assets/Reference_1min.wav')
musicUnb, sample_rate_unb = music_parser.readMusicFile('assets/unbek_2min.wav')

duration_song = librosa.get_duration(musicUnb)
# duration_window = 
print(duration_song)
frame_length =(duration_song * sample_rate_unb)
hopsize = 512
fs1 = sample_rate_unb/hopsize
fs2 = sample_rate_unb/hopsize

# Original anhand von JSON aufsplitten
    # split_segments = music_parser.splitReferenceRecording(t1.segments, sample_rate_ref, musicRef)

# CENS von jedem Segment erstellen (auf 1 fokussieren)
    # chromagrams_seg = music_parser.compute_chromagrams(split_segments, sample_rate_ref, tuning= 0, norm= 1, hop_length= hopsize, n_fft= frame_length)
    # print(chromagrams_seg)
    # CENS_ref = [] 
    # CENS_ref = compute_CENS_from_chromagrams_seg(chromagrams_seg, fs1)
    # print(CENS_ref)

C_ref = music_parser.compute_one_chromagram(musicRef, sample_rate_ref, tuning= 0, norm= 1, hop_length= hopsize, n_fft= frame_length)
CENS_ref, SR_ref = compute_CENS_from_chromagram(C_ref, fs1)

C_unb = music_parser.compute_one_chromagram(musicUnb, sample_rate_unb, tuning= 0, norm= 1, hop_length= hopsize, n_fft= frame_length)
CENS_unb, SR_unb = compute_CENS_from_chromagram(C_unb, fs2)

# CENS Reference Recording
plt.figure(figsize=(15, 5))
librosa.display.specshow(CENS_ref, x_axis='time', y_axis='chroma', sr= SR_ref, hop_length= hopsize, cmap= 'gray_r')
plt.title('Segment 1')
plt.colorbar()
plt.tight_layout()
plt.show()

# CENS Unknown Recording
    # plt.figure(figsize=(15, 5))
    # librosa.display.specshow(CENS_unb, x_axis='time', y_axis='chroma', sr= SR_unb, hop_length= hopsize, cmap= 'gray_r')
    # plt.title('CENS Unknown Recording')
    # plt.colorbar()
    # plt.tight_layout()
    # plt.show()
# unbekannte Version (verkürzt) in CENS 

# Segment als Querry über unbekannte Version jagen
