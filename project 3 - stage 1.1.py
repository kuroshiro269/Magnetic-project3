import numpy as np
import matplotlib.pyplot as plt

# Constants
index = 257698 * 2 # Hz
Fs = 10e6  # Sampling frequency Hz
f = index  # Frequency of the sinus wave (proportional to the index)
t = np.arange(0, 1 / f + 1 / Fs, 1 / Fs)  # Time vector
a = 1  # Amplitude
pfi = 0  # Phase offset
nfft = int(Fs)

x = a * np.sin(2 * np.pi * t * f + pfi)

# FFT
X = np.fft.fft(x, nfft)
X = X[:nfft // 2]
mx = np.abs(X) / np.max(np.abs(X))
frequencies = np.linspace(0, Fs / 2, nfft // 2)

# Generate the plot, title, and labels
plt.subplot(2, 1, 1)
plt.plot(t, x)
plt.title('Sine Wave Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.subplot(2, 1, 2)
plt.plot(frequencies, mx)
plt.title('Power Spectrum of a Sine Wave')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')

plt.tight_layout()
plt.show()
