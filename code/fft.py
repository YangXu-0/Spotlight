import numpy as np
import matplotlib.pyplot as plt
import librosa
import scipy

################ this is all pretty much just playing around with librosa and fft ################

def plot_audio_time_series(series, title):
    plt.figure(figsize=(18,5))

    time = np.linspace(0, len(series), len(series)) 
    plt.plot(time, series)
    plt.xlabel("Audio Time Series")
    plt.title(title)

    plt.show()


def plot_magnitude_spectrum(signal, title, sr):
    ft = np.fft.fft(signal)
    magnitude_spectrum = np.abs(ft)

    # plot magnitude spectrum
    plt.figure(figsize=(18, 5))

    frequency = np.linspace(0, sr, len(magnitude_spectrum))
    plt.plot(frequency, magnitude_spectrum)
    plt.xlabel("Frequency (Hz)")
    plt.title(title)

    plt.show()


##### Load audio time series using Librosa
audio_dir = "../assets/khan_noisy.wav"
violin, sr = librosa.load(audio_dir, sr=48000) # list of amplitudes and sample rate

print(violin, violin.shape)
print(sr)

#plot_audio_time_series(violin, "Violin Audio Time Series")

##### Process audio time series data using numpy
violin_ft = np.fft.fft(violin)

# same number of frequencies as the number of entries in time domain
# complex numbers --> I think it's info on both magnitude and frequency
#print(violin_ft, violin_ft.shape) 

magnitude_spectrum = np.abs(violin_ft)

#plot_magnitude_spectrum(violin, "Violin FFT", sr)


##### Deleting some frequencies
violin_ft = np.concatenate((violin_ft[0:3400]*0, violin_ft[3400:370000], violin_ft[370000:]*0))
print(violin_ft)

magnitude_spectrum = np.abs(violin_ft)
plt.figure(figsize=(18, 5))

frequency = np.linspace(0, sr, len(magnitude_spectrum))
plt.plot(frequency, magnitude_spectrum)
plt.xlabel("Frequency (Hz)")

plt.show()

##### Inverting the FFT to reproduce sound
violin_ift = np.real_if_close(np.fft.ifft(violin_ft)) # IFFT returns a complex due to floating precision number calc error
print(violin_ift, violin_ift.shape)
#plot_audio_time_series(violin_ift, "Violin Inverted Audio Time Series")

scipy.io.wavfile.write("../assets/inversed.wav", sr, violin_ift.astype(np.float32)) # Need to save as float32

violin, sr = librosa.load("../assets/inversed.wav")
#plot_magnitude_spectrum(violin, "Violin FFT", sr)

# I think the frequencies are doubled