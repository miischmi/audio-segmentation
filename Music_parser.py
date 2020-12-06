import librosa

class Music_parser():
    def __init__(self):
        pass

    def readMusicFile(self, path):
        y, sr = librosa.load(path)
        print(y)
        print(sr)