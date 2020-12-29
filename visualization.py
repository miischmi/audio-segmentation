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
    plt.xlabel('Time (minutes)')
    plt.yticks(np.arange(12), 'C C# D D# E F F# G G# A A# B'.split())
    plt.colorbar()
    plt.tight_layout()

def plot_matches(matches, Delta, Fs=1, alpha=0.2, color='r', s_marker='o', t_marker=''):
    """Plots matches into existing axis

    From: FMP-Notebooks, Müller & Zalkow (2019); Notebook: C7/C7S2_DiagonalMatching.ipynb

    Args:
        ax: Axis
        matches: Array of matches (start, end)
        alpha: Transparency pramaeter for match visualization
        color: Color used to indicated matches
        s_marker, t_marker: Marker used to indicate start and end of matches
    """
    axes = plt.gca()
    y_min, y_max = axes.get_yim()
    for (s, t) in matches:
        axes.plot(s/Fs, Delta[s], color=color, marker=s_marker, linestyle='None')
        axes.plot(t/Fs, Delta[t], color=color, marker=t_marker, linestyle='None')
        rect = patches.Rectangle(((s-0.5)/Fs, y_min), (t-s+1)/Fs, y_max, facecolor=color, alpha=alpha)
        axes.add_patch(rect) 

def plot_a_costmatrix(D, P, cmap= 'gray_r'):
    """Plots accumulated cost matrix with optimal warping path

    Args:
        D: accumulated cost matrix
        P: optimal warping path
        cmap: colormapping
    """
    P = np.array(P) 
    plt.figure(figsize=(90, 30))
    plt.imshow(D, cmap= cmap, origin= 'lower', aspect= 'equal')
    plt.plot(P[:, 1], P[:, 0], marker='o', color='r')
    plt.clim([0, np.max(D)])
    plt.colorbar()
    plt.title('$D$ with optimal warping path')
    plt.xlabel('Sequence Y')
    plt.ylabel('Sequence X')    

def plot_delta(Delta, matches):
    """Plots matching function with matches

    Args:
        Delta: matching function
        matches: Array containing matches (start, end)
    """
    plt.figure(figsize=(8, 4))
    plt.plot(Delta)
    plt.ylim(0, 1)
    plot_matches(matches, Delta, s_marker= '', t_marker= 'o')
    plt.tight_layout()
