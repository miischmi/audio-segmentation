import numpy as np
import librosa
import Music_parser as music_parser

def compute_local_average(x, M):
    """Compute local average of signal

    Notebook: C6/C6S1_NoveltySpectral.ipynb

    Args:
        x: Signal
        M: Determines size (2M+1) in samples of centric window  used for local average

    Returns:
        local_average: Local average signal
    """
    L = len(x)
    local_average = np.zeros(L)
    for m in range(L):
        a = max(m - M, 0)
        b = min(m + M + 1, L)
        local_average[m] = (1 / (2 * M + 1)) * np.sum(x[a:b])
    return local_average

def compute_novelty_spectrum(x, Fs=1, N=1024, H=256, gamma=100, M=10, norm=1):
    """Compute spectral-based novelty function

    Notebook: C6/C6S1_NoveltySpectral.ipynb

    Args:
        x: Signal
        Fs: Sampling rate
        N: Window size
        H: Hope size
        gamma: Parameter for logarithmic compression
        M: Size (frames) of local average
        norm: Apply max norm (if norm==1)

    Returns:
        novelty_spectrum: Energy-based novelty function
        Fs_feature: Feature rate
    """
    X = x
    Fs_feature = Fs / H
    Y = np.log(1 + gamma * np.abs(X))
    Y_diff = np.diff(Y)
    Y_diff[Y_diff < 0] = 0
    novelty_spectrum = np.sum(Y_diff, axis=0)
    novelty_spectrum = np.concatenate((novelty_spectrum, np.array([0.0])))
    if M > 0:
        local_average = compute_local_average(novelty_spectrum, M)
        novelty_spectrum = novelty_spectrum - local_average
        novelty_spectrum[novelty_spectrum < 0] = 0.0
    if norm == 1:
        max_value = max(novelty_spectrum)
        if max_value > 0:
            novelty_spectrum = novelty_spectrum / max_value
    return novelty_spectrum, Fs_feature

def get_chromagram(recording, sr, frame_length, hopsize, stft = False, **kwargs):
    tuning = kwargs.get('tuning', 0)
    norm = kwargs.get('norm', None)
    if stft:
        window = kwargs.get('window', None)
        return music_parser.compute_one_chromagram(recording, sr, norm=norm, hop_length=hopsize, n_fft=frame_length, window=window, tuning=tuning)
    else:
        midi = kwargs.get('herz', 21)
        flayout = kwargs.get('flayout', 'sos')
        bins_per_octave = kwargs.get('bins_per_octave', 12)
        n_octaves = kwargs.get('n_octaves', 7)
        center_freqs, sample_rates = music_parser.mr_frequencies_A0(tuning=tuning)
        time_freq = librosa.iirt(recording, sr=sr, win_length= frame_length, hop_length= hopsize, flayout = flayout, center_freqs=center_freqs, sample_rates=sample_rates)
        return librosa.feature.chroma_cqt(C=time_freq, bins_per_octave=bins_per_octave, n_octaves=n_octaves, fmin=librosa.midi_to_hz(midi), norm=norm)