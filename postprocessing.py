import numpy as np
import librosa
from scipy import signal



def normalize_feature_sequence(X, norm='2', threshold=0.0001, v=None):
    """Normalizes the columns of a feature sequence

     From: FMP-Notebooks, Müller & Zalkow (2019); Notebook: C3/C3S1_FeatureNormalization.ipynb

    Args:
        X: Feature sequence
        norm: The norm to be applied. '1', '2', 'max' or 'z'
        threshold: An threshold below which the vector `v` used instead of normalization
        v: Used instead of normalization below `threshold`. If None, uses unit vector for given norm

    Returns:
        X_norm: Normalized feature sequence
    """
    K,frame_length = X.shape
    X_norm = np.zeros((K,frame_length))
    if norm == '1':
        if v is None:
            v = np.ones(K) / K 
        for n in range(frame_length):
            s = np.sum(np.abs(X[:, n]))
            if s > threshold:
                X_norm[:, n] = X[:, n] / s
            else:
                X_norm[:, n] = v
    if norm == '2':
        if v is None:
            v = np.ones(K) / np.sqrt(K)
        for n in range(frame_length):
            s = np.sqrt(np.sum(X[:, n] ** 2))
            if s > threshold:
                X_norm[:, n] = X[:, n] / s
            else:
                X_norm[:, n] = v
    if norm == 'max':
        if v is None:
            v = np.ones(K)
        for n in range(frame_length):
            s = np.max(np.abs(X[:, n]))
            if s > threshold:
                X_norm[:, n] = X[:, n] / s
            else:
                X_norm[:, n] = v
    if norm == 'z':
        if v is None:
            v = np.zeros(K)
        for n in range(frame_length):
            mu = np.sum(X[:, n]) / K
            sigma = np.sqrt(np.sum((X[:, n] - mu) ** 2) / (K - 1))
            if sigma > threshold:
                X_norm[:, n] = (X[:, n] - mu) / sigma
            else:
                X_norm[:, n] = v
    return X_norm

def quantize_matrix(C, quant_fct=None):
    """Quantize matrix values in a logarithmic manner (as done for CENS features)

    From: FMP-Notebooks, Müller & Zalkow (2019); Notebook: C7/C7S2_CENS.ipynb

    Args:
        C: Input matrix
        quant_fct: List specifying the quantization function

    Returns:
        C_quant: Output matrix
    """
    C_quant = np.empty_like(C)
    if quant_fct is None:
        quant_fct = [(0.0, 0.05, 0), (0.05, 0.1, 1), (0.1, 0.2, 2), (0.2, 0.4, 3), (0.4, 1, 4)]
    for min_val, max_val, target_val in quant_fct:
        mask = np.logical_and(min_val <= C, C < max_val)
        C_quant[mask] = target_val
    return C_quant
    
def CENS_downsampling(X, Fs, filt_len=41, down_sampling=10, w_type='boxcar'):
    """Smoothes and downsamples a feature sequence. Smoothing is achieved by convolution with a filter kernel

    From: FMP-Notebooks, Müller & Zalkow (2019); Notebook: C3/C3S1_FeatureSmoothing.ipynb

    Args:
        X: Feature sequence
        Fs: Frame rate of `X`
        filt_len: Length of smoothing filter
        down_sampling: Downsampling factor
        w_type: Window type of smoothing filter

    Returns:
        X_smooth: Smoothed and downsampled feature sequence
        Fs_feature: Frame rate of `X_smooth`
    """
    filt_kernel = np.expand_dims(signal.get_window(w_type, filt_len), axis=0)
    X_smooth = signal.convolve(X, filt_kernel, mode='same') / filt_len
    X_smooth = X_smooth[:, ::down_sampling]
    Fs_feature = Fs / down_sampling
    return X_smooth, Fs_feature

def compute_CENS_from_chromagram(C, Fs= 1, ell= 41, d= 10, quant= True):
    """Compute CENS features from chromagram

    From: FMP-Notebooks, Müller & Zalkow (2019); Notebook: C7/C7S2_CENS.ipynb

    Args:
        C: Input chromagram
        Fs: Feature rate of chromagram
        ell: Smoothing length
        d: Downsampling factor
        quant: Apply quantization

    Returns:
        C_CENS: CENS features
        F_CENS: Feature rate of CENS features
    """
    C_norm = normalize_feature_sequence(C, norm='1')
    C_Q = quantize_matrix(C_norm) if quant else C_norm
    C_smooth, Fs_CENS = CENS_downsampling(C_Q, Fs, filt_len= ell, down_sampling= d, w_type= 'hann')
    C_CENS = normalize_feature_sequence(C_smooth, norm='2')

    return C_CENS, Fs_CENS

def compute_CENS_from_chromagrams_seg(segments, Fs= 1, ell=41, d=10, quant= True):
    """Compute CENS features from an array of chromagrams

    Adapted from: FMP-Notebooks, Müller & Zalkow (2019); Notebook: C7/C7S2_CENS.ipynb

    Args:
        segments: array of chromagrams
        Fs: Feature rate of chromagrams
        ell: Smoothing length
        d: Downsampling factor
        quant: Apply quantization

    Returns:
        cens_array: array of CENS feaatures
    """
    cens_array = []
    for C in segments:
        C_CENS, Fs_CENS = compute_CENS_from_chromagram(C, Fs, ell= ell, d= d)
        cens_array.append(C_CENS)
    return cens_array, Fs_CENS

def cyclic_shift(C, shift=1):
    """Cyclically shift a chromagram. Shifts a chroma vector by a certain number of semitones. 
    In other words, the chroma band D♯ becomes the chroma band E, the E becomes F and so on, and the last band B becomes C.

    From: FMP-Notebooks, Müller & Zalkow (2019); Notebook: C3/C3S1_TranspositionTuning.ipynb

    Args:
        C (np.ndarray): Chromagram
        shift (int): Tranposition shift (Default value = 1)

    Returns:
        C_shift (np.ndarray): Cyclically shifted chromagram
    """
    C_shift = np.roll(C, shift=shift, axis=0)
    return C_shift

def relativeOverlap(a_start, a_end, b_start, b_end):  
    """ computes Jaccard index for two Quantities

    From: Stuber (2021)
    
    Args:
        a_start: start of the computed segment in seconds
        a_end: end of the computed segment in seconds
        b_start: start of the reference segment in seconds
        b_end: end of the reference segment in seconds
    
    Returns:
        Relative overlap in percent (0.8 = 80%)
    """  
    if b_start > a_end or a_start > b_end:
        intersection = 0.00
    else:
        intersection = min(a_end, b_end) - max(a_start, b_start)
    
    union = (a_end - a_start) + (b_end - b_start) - intersection
        
    return intersection / union, intersection
        