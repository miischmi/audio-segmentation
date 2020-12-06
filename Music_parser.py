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
        print(len(recording_segments))
        print(len(recording_segments[0]))
        return recording_segments

    def chroma_censes(self, segments, sample_rate):
        chroma_cens_segments = []
        for segment in segments:
            chroma_cens_segments.append(librosa.feature.chroma_cens(segment, sample_rate))
        return chroma_cens_segments