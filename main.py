from JSON_Classifier import JSON_Classifier
from Music_parser import Music_parser
from Chroma_postprocessing import compute_CENS_from_chromagram
from Chroma_postprocessing import compute_CENS_from_chromagrams_seg
import datetime
import librosa
import librosa.display
import matplotlib.pyplot as plt


# Importing audio files
music_parser = Music_parser()
ref_recording, ref_samplerate = music_parser.readMusicFile('assets/Reference_1min.wav')
test_recording, test_samplerate = music_parser.readMusicFile('assets/unbek_2min.wav')

# Importing the metadata from the JSON-file
meta_data = JSON_Classifier()
meta_data.readJSON('assets/testdata.json')

# Splitting the audio file in segments, according to the metadata
segment_list = music_parser.splitReferenceRecording(meta_data.segments, ref_samplerate, ref_recording,)

# Feature Extraction
ref_length = librosa.get_duration(ref_recording)
test_length = librosa.get_duration(test_recording)
frame_length = 2048
hopsize = 512
window = 'hann'
test_featurerate = None
ref_featurerate = None

# Creating each chromagram
ref_chromagram = music_parser.compute_chromagrams(segment_list, ref_samplerate, norm= 1, hop_length= hopsize, n_fft= frame_length, window = window)
test_chromagram = music_parser.compute_one_chromagram(test_recording, test_samplerate, norm= 1, hop_length= hopsize, n_fft= frame_length, window = window)

# Creating CENS Test-recording
CENS_test, CENS_test_featurerate = compute_CENS_from_chromagram(test_chromagram, test_featurerate)
plt.figure(figsize=(15, 5))
librosa.display.specshow(CENS_test, x_axis='time', y_axis='chroma', sr= CENS_test_featurerate, hop_length= hopsize, cmap= 'gray_r')
plt.title('Test Recording')
plt.colorbar()
plt.tight_layout()
plt.show()

# Creating CENS Reference Recording
CENS_ref = compute_CENS_from_chromagrams_seg(ref_chromagram, ref_featurerate)

# Segment als Querry über unbekannte Version jagen
