import librosa
import numpy as np


    
def readMusicFile(path):
    return librosa.load(path, sr= None)

def splitReferenceRecording(segments, sample_rate, recording):
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

def compute_chromagrams(segments, sample_rate, norm = None, hop_length = 512, n_fft = 2048, window = 'hann', tuning = None):
    """Computation of a chromagram for each segment in an array of segments

    Args:
        segments: array of segments
        rest: see documentation librosa.feature.chroma_stft
            
    Returns: 
        segments_chromagrams: array of chromagrams
    """ 
    segments_chromagrams = []
    for segment in segments:
        segments_chromagrams.append(compute_one_chromagram(segment, sample_rate, norm= norm, hop_length= hop_length, n_fft= n_fft, window = window, tuning= tuning))
    return segments_chromagrams

def compute_one_chromagram(music, sample_rate, norm = None, hop_length = 512, n_fft = 2048, window = 'hann', tuning = None):
    return librosa.feature.chroma_stft(y= music, sr= sample_rate, norm= norm, hop_length= hop_length, n_fft= n_fft, window = window, tuning= tuning)

def compute_power_db(x, sr, win_len_sec=0.1, power_ref=10**(-12)):
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

def mr_frequencies_A0(tuning):
    r'''Helper function for generating center frequency and sample rate pairs.
    Adapted from Librosa mr_frequencies()

    This function will return center frequency and corresponding sample rates
    to obtain similar pitch filterbank settings as described in [1]_.
    Starting at A0.

    .. [1] Müller, Meinard.
        "Information Retrieval for Music and Motion."
        Springer Verlag. 2007.


    Parameters
    ----------
    tuning : float in `[-0.5, +0.5)` [scalar]
        Tuning deviation from A440, measure as a fraction of the equally
        tempered semitone (1/12 of an octave).

    Returns
    -------
    center_freqs : np.ndarray [shape=(n,), dtype=float]
        Center frequencies of the filter kernels.
        Also defines the number of filters in the filterbank.

    sample_rates : np.ndarray [shape=(n,), dtype=float]
        Sample rate for each filter, used for multirate filterbank.

    Notes
    -----
    This function caches at level 10.


    See Also
    --------
    librosa.filters.semitone_filterbank
    librosa.filters._multirate_fb
    '''

    center_freqs = librosa.midi_to_hz(np.arange(21 + tuning, 109 + tuning))

    sample_rates = np.asarray(len(np.arange(0, 36)) * [882, ] +
                            len(np.arange(36, 72)) * [4410, ] +
                            len(np.arange(72, 88)) * [22050, ])

    return center_freqs, sample_rates

