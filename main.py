from JSON_Classifier import JSON_Classifier
from Music_parser import Music_parser
import datetime

t1 = JSON_Classifier()
t1.readJSON('assets/testdata.json')
music_parser = Music_parser()
music_parser.readMusicFile('assets/Testsegmente_1_6.wav')

