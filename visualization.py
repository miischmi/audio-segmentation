import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

def plot_waveform(t, recording, title = None):
    """Plots Waveform of a Signal

    Args:
        t: Time axis (in seconds)
        recording: Input signal
        title: Title of the plot
    """
    plt.figure(figsize=(10, 2))
    plt.plot(t, recording, color='gray')
    plt.title(title)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.tick_params(direction='in')
    plt.tight_layout()

def plot_powerdb(t, power_db, title= None):
    """Plots Power level over time

    Args:
        t: Time axis (in seconds)
        recording: Input signal
        title: Title of the plot
    """
    plt.figure(figsize=(10, 2))
    plt.plot(t, power_db, color='red')
    plt.title(title)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Power (dB)')
    plt.ylim(40,100)
    plt.tick_params(direction='in')
    plt.tight_layout()

def plot_chromagram(chromagram, sr=48000, hopsize= 4800, title= None, cmap = 'gray_r', clim = [0,1]):
    """Plots Chromagram

    Args:
        chromagram: Chromagram
        sr: sampling rate of Audio
        hopsize: hopsize
        title: Title of the plot
        cmap: colormapping
        clim: colorbar size
    """    
    plt.figure(figsize=(15, 5))
    librosa.display.specshow(chromagram , x_axis='time', y_axis='chroma', sr= sr, hop_length= hopsize, cmap= cmap, clim= clim)
    plt.title(title)
    plt.xlabel('Time (seconds)')
    plt.yticks(np.arange(12), 'C C# D D# E F F# G G# A A# B'.split())
    plt.colorbar()
    plt.tight_layout()

def plot_CENS(cens, fs= 48000, hopsize= 4800, title= None, cmap = 'gray_r', clim = [0,1]):
    """Plots CENS feature

    Args:
        cens: CENS
        fr: sampling rate of CENS feature
        hopsize: hopsize
        title: Title of the plot
        cmap: colormapping
        clim: colorbar size
    """      
    plt.figure(figsize=(15, 5))
    librosa.display.specshow(cens , x_axis='time', y_axis='chroma', sr= fs, hop_length= hopsize, cmap= cmap, clim= clim )
    plt.title(title)
    plt.xlabel('Time (seconds)')
    plt.yticks(np.arange(12), 'C C# D D# E F F# G G# A A# B'.split())
    plt.colorbar()
    plt.tight_layout()

def plot_matches(ax, matches, Delta, Fs=1, alpha=0.2, color='r', s_marker='o', t_marker=''):
    """Plots matches into existing axis

    From: FMP-Notebooks, Müller & Zalkow (2019); Notebook: C7/C7S2_DiagonalMatching.ipynb

    Args:
        ax: Axis
        matches: Array of matches (start, end)
        alpha: Transparency pramaeter for match visualization
        color: Color used to indicated matches
        s_marker, t_marker: Marker used to indicate start and end of matches
    """
    y_min, y_max = ax.get_ylim()
    for (s, t) in matches:
        ax.plot(s/Fs, Delta[s], color=color, marker=s_marker, linestyle='None')
        ax.plot(t/Fs, Delta[t], color=color, marker=t_marker, linestyle='None')
        rect = patches.Rectangle(((s-0.5)/Fs, y_min), (t-s+1)/Fs, y_max, facecolor=color, alpha=alpha)
        ax.add_patch(rect) 

def plot_accCostMatrix_and_Delta(D, P, Delta, matches, ax, ref_track, test_track, segment, Fs= 1, cmap= 'gray_r'):
    """Plots accumulated cost matrix with optimal warping path and matching function with matches

    Args:
        D: accumulated cost matrix
        P: optimal warping path
        Delta: matching function
        matches: Array containing matches (start, end)
        ax: Axis
        ref_track, test_track: Names of the .wav files
        segment: Segment number
        cmap: colormapping
    """
    P = np.array(P) 
    ax[0].imshow(D, cmap= cmap, origin= 'lower', aspect= 'auto')
    ax[0].plot(P[:, 1], P[:, 0], marker='o', color='r')
    ax[0].set_title('$D$ with optimal warping path')
    ax[0].set_xlabel('Sequence Y')
    ax[0].set_ylabel('Sequence X')  

    ax[1].plot(Delta)
    ax[1].set_ylim(0, 0.6)
    plot_matches(ax= ax[1], matches= matches, Delta= Delta, Fs= Fs, s_marker= '', t_marker= 'o')
    ax[1].set_title(r'Matching function $\Delta$')
    ax[1].set_xlabel('Time (samples)')
    plt.suptitle(f'Recordings: {ref_track} vs. {test_track}, Segment: {segment}')

def plot_costmatrix(C, Fs= 9600, hopsize = 9600, cmap= 'gray_r'):
    """Plots accumulated cost matrix with optimal warping path

    Args:
        c:  cost matrix
        cmap: colormapping
    """
    N, M = C.shape
    H= hopsize
    left = - (N / Fs) / 2
    right = C.shape[1] * H / Fs + (N / Fs) / 2
    lower = - (N / Fs) / 2
    upper = C.shape[0] * H / Fs + (N / Fs) / 2
    plt.imshow(C, cmap=cmap, aspect='auto', origin='lower', extent=[left, right, lower, upper])
    plt.xlabel('Time (seconds)')
    plt.ylabel('Time (seconds)')
    plt.tight_layout()  