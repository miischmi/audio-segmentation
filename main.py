from JSON_Classifier import JSON_Classifier
from Music_parser import Music_parser
import datetime
import librosa
import librosa.display
import matplotlib.pyplot as plt

t1 = JSON_Classifier()
t1.readJSON('assets/testdata.json')
print(t1.segments)
music_parser = Music_parser()
music, sample_rate = music_parser.readMusicFile('assets/Testsegmente_1_6.wav')

# Original anhand von JSON aufsplitten
split_segments = music_parser.splitReferenceRecording(t1.segments, sample_rate, music)

# CENS von jedem Segment erstellen (auf 1 fokussieren)
chroma_censes = music_parser.chroma_censes(split_segments, sample_rate)
print(len(chroma_censes))

# unbekannte Version (verkürzt) in CENS 

# Segment als Querry über unbekannte Version jagen
