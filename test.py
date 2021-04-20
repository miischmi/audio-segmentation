import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import librosa
from Music_parser import Music_parser

music_parser = Music_parser()

center_freqs, sample_rates = music_parser.mr_frequencies_A0(tuning=0.0)
semitone_filterbank, sample_rates = librosa.filters.semitone_filterbank(center_freqs=center_freqs, sample_rates=sample_rates)

fig, ax = plt.subplots(figsize=(15,3))
for cur_sr, cur_filter in zip(sample_rates[72:88], semitone_filterbank):
   w, h = scipy.signal.freqz(cur_filter[0], cur_filter[1], worN=2000)
   ax.semilogx((cur_sr / (2 * np.pi)) * w, 20 * np.log10(abs(h)))
ax.set(xlim=[20, 10e3], ylim=[-60, 3], title= r'Filters for $\mathit{p}$ $\in$ [96:108] with respect to $\mathit{22050 Hz}$', xlabel='Log-Frequency (Hz)', ylabel='Magnitude (dB)')
plt.show()