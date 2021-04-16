import numpy as np
import scipy
from  matplotlib import pyplot as plt
import librosa
import librosa.display


duration = 1.0
Fs = 4000
N = int(duration * Fs)
t = np.arange(N) / Fs
x = np.sin(2 * np.pi * 400 * t) + np.sin(2 * np.pi * 450 * t)
x[int(round(0.45 * Fs))] = 10
x[int(round(0.50 * Fs))] = 10


plt.figure(figsize=(6.5, 1.5))
plt.plot(t, x, c='gray')
plt.xlabel('Time (seconds)')
plt.xlim( [t[0], t[-1]] )
plt.title('Signal')
plt.tight_layout()
plt.show()

w_len_ms = 32
N = int((w_len_ms / 1000) * Fs)
H = 16
X_short = librosa.stft(x, n_fft=N*16, hop_length=H, win_length=N, window='hann', center=True, pad_mode='constant')

w_len_ms = 128
N = int((w_len_ms / 1000) * Fs)
H = 16
X_long = librosa.stft(x, n_fft=N*16, hop_length=H, win_length=N, window='hann', center=True, pad_mode='constant')

plt.figure(figsize=(8, 3))
librosa.display.specshow(librosa.amplitude_to_db(np.abs(X_short), ref=np.max), y_axis='linear', x_axis='time', sr=Fs, hop_length=H, cmap='gray_r')
plt.clim([-90, 0])
plt.ylim([0, 1000])
plt.xlim([0, 1])
plt.colorbar(format='%+2.0f dB')
plt.xlabel('Time (seconds)')
plt.ylabel('Frequency (Hz)')
plt.title('Short Hann window')
plt.tight_layout()

plt.figure(figsize=(8, 3))
librosa.display.specshow(librosa.amplitude_to_db( np.abs(X_long), ref=np.max), y_axis='linear', x_axis='time', sr=Fs, hop_length=H, cmap='gray_r')
plt.clim([-90, 0])
plt.ylim([0, 1000])
plt.xlim([0, 1])
plt.colorbar(format='%+2.0f dB')
plt.xlabel('Time (seconds)')
plt.ylabel('Frequency (Hz)')
plt.title('Long Hann window')
plt.tight_layout()
plt.show()

