import numpy as np
import matplotlib.pyplot as plt

# Constants
A = 1.0  # Amplitude
K = 1.0  # Scaling factor
f = 257698  # Frequency
phi_values = [0, np.pi / 4, np.pi / 2]  # Phase angle
t1_values = [0.1, 0.2, 0.3]  # Time constant 1
t2_values = [0.5, 0.8, 1.0]  # Time constant 2
n_values = [1, 2, 3]  # Slope of the decaying sinusoid characteristic

# Determine the maximum number of iterations
num_iter = min(len(phi_values), len(t1_values), len(t2_values), len(n_values))
num_plots = num_iter * num_iter  # Number of subplots required

# Time domain analysis
t = np.linspace(0, 2, 1000)  # Time values for analysis

plt.figure(figsize=(12, 8))

for i in range(num_plots):
    phi_idx = i // num_iter  # Index for phi_values
    t1_idx = (i // num_iter ** 2) % num_iter  # Index for t1_values
    t2_idx = (i // num_iter) % num_iter  # Index for t2_values
    n_idx = i % num_iter  # Index for n_values

    phi = phi_values[phi_idx]
    t1 = t1_values[t1_idx]
    t2 = t2_values[t2_idx]
    n = n_values[n_idx]

    # Generate the signal
    w = A * K * (((t / t1) ** n) / (1 + ((t / t1) ** n))) * np.exp(-t / t2) * np.cos(2 * np.pi * f * t + phi)

    # Plot the signal in the time domain
    plt.subplot(num_iter, num_iter, i + 1)
    plt.plot(t, w)
    plt.title(f"phi={phi:.2f}, t1={t1:.2f}, t2={t2:.2f}, n={n}")

plt.suptitle("Time Domain Analysis")
plt.tight_layout()
plt.show()

# Frequency domain analysis
Fs = 10000000 # Sampling frequency
N = len(t)  # Number of samples
frequencies = np.fft.fftfreq(N, 1 / Fs)

plt.figure(figsize=(12, 8))

for i in range(num_plots):
    phi_idx = i // num_iter  # Index for phi_values
    t1_idx = (i // num_iter ** 2) % num_iter  # Index for t1_values
    t2_idx = (i // num_iter) % num_iter  # Index for t2_values
    n_idx = i % num_iter  # Index for n_values

    phi = phi_values[phi_idx]
    t1 = t1_values[t1_idx]
    t2 = t2_values[t2_idx]
    n = n_values[n_idx]

    # Generate the signal
    w = A * K * (((t / t1) ** n) / (1 + ((t / t1) ** n))) * np.exp(-t / t2) * np.cos(2 * np.pi * f * t + phi)

    # Compute the FFT
    spectrum = np.abs(np.fft.fft(w))

    # Plot the spectrum
    plt.subplot(num_iter, num_iter, i + 1)
    plt.plot(frequencies[:N//2], spectrum[:N//2])
    plt.title(f"phi={phi:.2f}, t1={t1:.2f}, t2={t2:.2f}, n={n}")

plt.suptitle("Frequency Domain Analysis")
plt.tight_layout()
plt.show()
