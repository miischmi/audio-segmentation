import librosa

class Music_parser():
    def __init__(self):
        pass

    def readMusicFile(self, path):
        return librosa.load(path)

    def splitReferenceRecording(self, segments, sample_rate, recording):
        recording_segments = []
        for segment in segments:
            start_split_at_second = segment['start']
            end_split_at_second = segment['ende']
            split_beginning= sample_rate * start_split_at_second
            split_ending = sample_rate * end_split_at_second
            recording_segments.append(recording[split_beginning:split_ending])
        # Debug prints
        # print(len(recording_segments))
        # print(len(recording_segments[0]))
        return recording_segments
    
    def compute_chromagrams(self, segments, sample_rate, norm = None, hop_length = 512, n_fft = 2048, window = 'hann'):
        segments_chromagrams = []
        for segment in segments:
            segments_chromagrams.append(librosa.feature.chroma_stft(y= segment, sr= sample_rate, norm= norm, hop_length= hop_length, n_fft= n_fft, window = window))
        return segments_chromagrams

    def compute_one_chromagram(self, music, sample_rate, norm = None, hop_length = 512, n_fft = 2048, window = 'hann'):
        return librosa.feature.chroma_stft(y= music, sr= sample_rate, norm= norm, hop_length= hop_length, n_fft= n_fft, window = window)

    



