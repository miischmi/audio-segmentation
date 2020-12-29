import librosa
import numpy as np

class Music_parser():
    
    def readMusicFile(self, path):
        return librosa.load(path, sr= None)

    def splitReferenceRecording(self, segments, sample_rate, recording):
        """Split a Recording in segments according to a given JSON file
                
        Returns: 
            recording_segments: array of arrays of librosa floating point time series
        """ 
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
        """Computation of a chromagram for each segment in an array of segments

        Args:
            segments: array of segments
            rest: see documentation librosa.feature.chroma_stft
                
        Returns: 
            segments_chromagrams: array of chromagrams
        """ 
        segments_chromagrams = []
        for segment in segments:
            segments_chromagrams.append(librosa.feature.chroma_stft(y= segment, sr= sample_rate, norm= norm, hop_length= hop_length, n_fft= n_fft, window = window))
        return segments_chromagrams

    def compute_one_chromagram(self, music, sample_rate, norm = None, hop_length = 512, n_fft = 2048, window = 'hann'):
        return librosa.feature.chroma_stft(y= music, sr= sample_rate, norm= norm, hop_length= hop_length, n_fft= n_fft, window = window)

    def compute_power_db(self, x, sr, win_len_sec=0.1, power_ref=10**(-12)):
        """Computation of the signal power in dB
        
        From: FMP-Notebooks, Müller & Zalkow (2019); Notebook: C1/C1S3_Dynamics.ipynb
        
        Args: 
            x: Signal (waveform) to be analyzed
            sr: Sampling rate
            win_len_sec: Length (seconds) of the window
            power_ref: Reference power level (0 dB)
        
        Returns: 
            power_db: Signal power in dB
        """     
        win_len = round(win_len_sec * sr)
        win = np.ones(win_len) / win_len
        power_db = 10 * np.log10(np.convolve(x**2, win, mode='same') / power_ref)    
        return power_db



